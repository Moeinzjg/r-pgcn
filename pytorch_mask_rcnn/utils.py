import os
import re
import torch
import cv2
import numpy as np


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
