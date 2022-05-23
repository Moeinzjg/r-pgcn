from __future__ import division
import os
import re
import torch
import cv2
import numpy as np
import scipy.optimize
import shapely

from lydorn_utils import polygon_utils


__all__ = ["save_ckpt", "Meter"]


def save_ckpt(model, optimizer, epochs, ckpt_path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(ckpt_path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, ckpt_path)
    
    
class TextArea:
    def __init__(self):
        self.buffer = []
    
    def write(self, s):
        self.buffer.append(s)
        
    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        result = {"bbox AP": 0.0, "mask AP": 0.0}
        
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        if len(values) > 0:
            values = [int(v) / 10 for v in values]
            result = {"bbox AP": values[0], "mask AP": values[12]}
            
        return result

   
class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)


class Matcher:
    def __init__(self, threshold):
        self.threshold = threshold

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

        label[value >= self.threshold] = 1
        label[value < self.threshold] = 0

        return label, matched_idx


def box_iou(box_a, box_b):
    """
    Arguments:
        boxe_a (Tensor[N, 4])
        boxe_b (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    
    return inter / (area_a[:, None] + area_b - inter)


def match_boxes(gt_box, pred_box, threshold=0.5):
    box_matcher = Matcher(threshold)
    iou = box_iou(gt_box, pred_box)
    pos_neg_label, matched_idx = box_matcher(iou)
    pass


def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))
    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou


def iou_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    masks = np.zeros((2, height, width), dtype=np.uint8)
    # import pdb; pdb.set_trace()
    if isinstance(pred, list):
        for p in pred:
            masks[0] = draw_poly12(masks[0], p)
    else:
        for idx in range(pred.shape[0]):
            masks[0] = draw_poly12(masks[0], pred[idx, :, :])

    if isinstance(gt, list):
        for g in gt:
            masks[1] = draw_poly12(masks[1], g)
    else:
        for idx in range(gt.shape[0]):
            masks[1] = draw_poly12(masks[1], gt[idx, :, :])
    return iou_from_mask(masks[0], masks[1]), masks


def mask_from_poly(pred, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    grid_size: grid_size that the polygons are in

    """
    if isinstance(pred, list):
        masks = np.zeros((len(pred), height, width), dtype=np.uint8)
        for idx, p in enumerate(pred):
            masks[idx] = draw_poly12(masks[idx], p)
    else:
        masks = np.zeros((pred.shape[0], height, width), dtype=np.uint8)
        for idx in range(pred.shape[0]):
            masks[idx] = draw_poly12(masks[idx], pred[idx, :, :])

    return masks


def draw_poly12(mask, poly, fill_value=255):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly.cpu())
    poly = np.int32(poly)

    cv2.fillPoly(mask, [poly], fill_value)

    return mask


#TODO: standardize the thresholds and evaluate on the same detections

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)
    idx_gt = idx_gt_actual[sel_valid]
    idx_pred = idx_pred_actual[sel_valid]
    bbox_gt = bbox_gt[idx_gt]

    area = [(bbox_gt[i, 2] - bbox_gt[i, 0] + 1) * (bbox_gt[i, 3] - bbox_gt[i, 1] + 1) for i in range(bbox_gt.shape[0])]
    slope = [(bbox_gt[i, 3] - bbox_gt[i, 1] + 1) / (bbox_gt[i, 2] - bbox_gt[i, 0] + 1) for i in range(bbox_gt.shape[0])]
    
    return idx_gt, idx_pred, ious_actual[sel_valid], label, area, slope


def compute_contour_metrics(poly_gt, poly_pred):
    gt_polygons = [shapely.geometry.Polygon(np.array(poly)) for poly in poly_gt]
    dt_polygons = [shapely.geometry.Polygon(np.array(poly)) for poly in poly_pred]

    fixed_gt_polygons = polygon_utils.fix_polygons(gt_polygons, buffer=0.0001)  # Buffer adds vertices but is needed to repair some geometries
    fixed_dt_polygons = polygon_utils.fix_polygons(dt_polygons)

    max_angle_diffs = polygon_utils.compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)
    return max_angle_diffs


def maxtan_from_poly(pred, gt):
    """
    Compute Max tangent from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predictions
    gt: list of np arrays of gt
    grid_size: grid_size that the polygons are in

    """
    bbox_gt = gt['boxes'].detach().cpu().numpy()
    bbox_pred = pred['boxes'].detach().cpu().numpy()  # TODO: filter out boxes with low score
    gt_idx, pred_idx, ious, label, area, slope = match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5)

    poly_gt = [gt['global_polygons'][i] for i in gt_idx]
    poly_pred = pred['polygons'].detach().cpu().numpy()
    poly_pred = [poly_pred[j] for j in pred_idx]


    _, masks = iou_from_poly(pred['polygons'].detach().cpu().numpy(), gt['global_polygons'], 650, 650)

    max_angle_diffs = compute_contour_metrics(poly_gt, poly_pred)
    poly_nvertex = [poly.shape[0] for poly in poly_gt]
    if len(max_angle_diffs) == 0:
        area = []
        slope = []
        poly_nvertex = []
    return max_angle_diffs, area, slope, poly_nvertex


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