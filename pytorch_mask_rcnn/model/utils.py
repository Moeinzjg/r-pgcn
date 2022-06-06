import torch
import numpy as np
import skimage


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """

        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) 

        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0

        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx


class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx

    
def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios

        self.cell_anchor = None
        self._cache = {}

    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1)
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        y, x = torch.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)

        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        return anchor

    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)

        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))

        self.set_cell_anchor(dtype, device)

        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor


def prepare_gcn_component(pred_polys, grid_sizes, max_poly_len, n_adj=3):

    batch_array_feature_indexs = []

    curr_p = pred_polys
    p_index = []
    for size in grid_sizes:
        curr_p_grid_size = np.floor(curr_p * size).astype(np.int32)

        curr_p_grid_size[:, 1] *= size
        curr_p_index = np.sum(curr_p_grid_size, axis=-1)
        p_index.append(curr_p_index)

    array_feature_indexs = np.zeros((len(grid_sizes), max_poly_len), np.float32)

    array_feature_indexs[:, :max_poly_len] = np.array(p_index)

    batch_array_feature_indexs.append(array_feature_indexs)

    adj_matrix = create_adjacency_matrix_cat(1, n_adj, max_poly_len)

    return {'feature_indexs': torch.Tensor(np.stack(batch_array_feature_indexs, axis=0)),
            'adj_matrix': torch.Tensor(adj_matrix)
            }


def create_adjacency_matrix_cat(batch_size, n_adj, n_nodes):
    a = np.zeros([batch_size, n_nodes, n_nodes])
    m = int(-n_adj / 2)
    n = int(n_adj / 2 + 1)

    for t in range(batch_size):
        for i in range(n_nodes):
            for j in range(m, n):
                if j != 0:
                    a[t][i][(i + j) % n_nodes] = 1
                    a[t][(i + j) % n_nodes][i] = 1

    return a.astype(np.float32)


def gather_feature(id, feature):
    feature_id = id.unsqueeze_(2).long().expand(id.size(0),
                                                id.size(1),
                                                feature.size(2)).detach()
    cnn_out = torch.FloatTensor()
    cnn_out = torch.gather(feature, 1, feature_id).float()

    return cnn_out


def get_initial_points(cp_num):
    pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
    for i in range(cp_num):
        thera = 1.0 * i / cp_num * 2 * np.pi - (np.pi / 4.0)
        if thera < 0:
            thera += 2 * np.pi
        if thera > 2 * np.pi:
            thera -= 2 * np.pi
        x = np.cos(thera)
        y = -np.sin(thera)
        pointsnp[i, 0] = x
        pointsnp[i, 1] = y

    fwd_poly = (0.7 * pointsnp + 1) / 2

    arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
    arr_fwd_poly[:, :] = fwd_poly
    return arr_fwd_poly


def sample_points_from_mask_contours(masks):
    '''computes the contour from segmentation mask; then samples from them.'''
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    batch_size = masks.shape[0]
    poly = []
    for i in range(batch_size):
        mask = masks[i, 1, :, :].detach().cpu().numpy()
        mask = np.uint8((mask > 0.5))
        # Using marching squares
        contours = skimage.measure.find_contours(mask, 0.5, fully_connected='low', positive_orientation='high')
        # Simplify contours a tiny bit using Douglas-Peucker algorithm:
        contours = [skimage.measure.approximate_polygon(contour, tolerance=0.01) for contour in contours]
        lengths = [contour.shape[0] for contour in contours]
        if len(lengths) > 0:
            longest_idx = np.array(lengths).argmax()
        # contours = np.concatenate(np.array(contours))
        poly.append(torch.from_numpy(np.array(contours[longest_idx])))  # get the longest one
    return poly


# def sample_points_from_edges(mask):
#     '''computes the contour from segmentation mask; then samples from them.'''
#     # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
#     batch_size = mask.shape[0]
#     for i in range(batch_size):
#         mask = mask[i, 1, :, :].detach().cpu().numpy()
#         import matplotlib.pyplot as plt
#         plt.imshow(mask)
#         plt.show()
#         mask = np.uint8(mask * 255)
#         contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

#         # draw contours on the original image
#         image_copy = mask.copy()
#         cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

#         # see the results
#         cv2.imshow('None approximation', image_copy)
#         cv2.waitKey(0)

#     return mask


    # pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
    # for i in range(cp_num):
    #     thera = 1.0 * i / cp_num * 2 * np.pi - (np.pi / 4.0)
    #     if thera < 0:
    #         thera += 2 * np.pi
    #     if thera > 2 * np.pi:
    #         thera -= 2 * np.pi
    #     x = np.cos(thera)
    #     y = -np.sin(thera)
    #     pointsnp[i, 0] = x
    #     pointsnp[i, 1] = y

    # fwd_poly = (0.7 * pointsnp + 1) / 2

    # arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
    # arr_fwd_poly[:, :] = fwd_poly
    # return arr_fwd_poly


def polis_metric(poly1, poly2):
    p1 = 0
    p2 = 0
    for j in range(poly1.shape[0]):
        p1 += min_dist_point2poly(poly1[j, :], poly2)
    for k in range(poly2.shape[0]):
        p2 += min_dist_point2poly(poly2[k, :], poly1)
    
    return (p1 / (2 * poly1.shape[0]) + p2 / (2 * poly2.shape[0])) / 2


def min_dist_point2poly(point, poly):
    '''
    AX+BY+C=0
    A = y1 - y2
    B = x2 - x1
    C = x1y2 - x2y1
    '''
    dis = torch.ones((poly.shape[0]), 1) * 100
    for i in range(poly.shape[0] - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i+1]
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        if torch.sqrt(A*A + B*B) > 0:
            dis[i] = torch.abs(A * point[0] + B * point[1] + C)/torch.sqrt(A*A + B*B)
    return dis.min()


# Losses useful for the polygon similarity loss and their associated functions
def nearest_vertex_matching(poly_fewer, poly_more):
    '''Returns the matching (poly_fewer -> poly_more) index which shows each
       of the vertices in poly_fewer matches which of the ones in poly_more. Thus, the
       size would be the same as poly_fewer in the first dimension.
    '''
    l2_dist = torch.cdist(poly_fewer, poly_more, p=2)
    matching_idx = torch.ones((poly_fewer.shape[0])) * 1000
    distance = 0.0
    for fewer_idx in range(poly_fewer.shape[0]):
        # iteratively remove the col and row corresponding to the min distance
        min_idx = l2_dist.argmin()
        min_row = min_idx // l2_dist.shape[1]
        min_col = min_idx - min_row * l2_dist.shape[1]
        matching_idx[fewer_idx] = min_col
        distance += l2_dist[min_row, min_col]
        l2_dist[min_row, :] = 100.0
        l2_dist[:, min_col] = 100.0

    return distance, matching_idx


def biprojection_loss(pred, gt):
    '''Measures the difference between predicted and ground truth polygon vertices.
       Assuming that we have two polygons Pm with more vertices and Pf with fewer ones.
       First, it finds the one-to-one matching for all the vertices in Pf, then computes the distance
       between each left vertex in Pm and the projection of that vertex on the closest edge in Pf.
    '''
    if pred.shape[0] <= gt.shape[0]:
        poly_fewer, poly_more = pred, gt
    else:
        poly_fewer, poly_more = gt, pred
    
    # find the matching for vertices in Pf
    distance, matching_idx = nearest_vertex_matching(poly_fewer, poly_more)

    # find the min distance of each unmatched vertex in Pm to the edges of Pm
    for idx in range(poly_more.shape[0]):
        if idx in matching_idx.tolist():
            continue
        distance += min_dist_point2poly(poly_more[idx, :], poly_fewer)  # TODO: check this function   #FIXME: vertex number in prediction is still 16
    distance /= poly_more.shape[0]
    return distance, matching_idx


def relative_shape_loss(pred, gt, matching_idx):
    ''' Penalizes the matched line segments which 
        are not parallel with each other. The matching
        is inherited from vertex matching in the bi-projection
        loss. The sine measure is calculated for directional 
        vectors of two matched segments.
    '''
    if pred.shape[0] <= gt.shape[0]:
        poly_fewer, poly_more = pred, gt
    else:
        poly_fewer, poly_more = gt, pred
    pass
    
    
