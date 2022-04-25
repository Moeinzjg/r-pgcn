import copy
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import shapely.geometry


import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from lydorn_utils import polygon_utils


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        #self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        
        self.has_results = False
            
    def accumulate(self, coco_results):  # input all predictions
        if len(coco_results) == 0:
            return
        
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = self.coco_gt.loadRes(coco_results)  # use the method loadRes
            coco_eval.params.imgIds = image_ids  # ids of images to be evaluated
            coco_eval.evaluate()  # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate()  # 3s
            
        self.has_results = True
    
    def summarize(self):
        if self.has_results:
            for iou_type in self.iou_types:
                print("IoU metric: {}".format(iou_type))
                self.coco_eval[iou_type].summarize()
            
        else:
            print("evaluation has no results")

    # def polygonal_metrics(self, coco_results_poly):
    #     coco_gt = self.coco_gt
    #     coco_dt = coco_gt.loadRes(coco_results_poly)
    #     contour_eval = ContourEval(coco_gt, coco_dt)
    #     max_angle_diffs = contour_eval.evaluate()
    #     import pdb; pdb.set_trace()


def prepare_for_coco(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]
        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        masks = masks > 0.5
        rles = [
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )
    return coco_results


def prepare_for_coco_polygon(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks_from_polygons"]
        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        masks = masks > 0.5
        rles = [
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )
    return coco_results


# def compute_contour_metrics(gts_dts):
#     gts, dts = gts_dts
#     gt_polygons = [shapely.geometry.Polygon(np.array(coords).reshape(-1, 2)) for ann in gts
#                    for coords in ann["segmentation"]]
#     dt_polygons = [shapely.geometry.Polygon(np.array(coords).reshape(-1, 2)) for ann in dts
#                    for coords in ann["segmentation"]]
#     fixed_gt_polygons = polygon_utils.fix_polygons(gt_polygons, buffer=0.0001)  # Buffer adds vertices but is needed to repair some geometries
#     fixed_dt_polygons = polygon_utils.fix_polygons(dt_polygons)
#     # cosine_similarities, edge_distances = \
#     #     polygon_utils.compute_polygon_contour_measures(dt_polygons, gt_polygons, sampling_spacing=2.0, min_precision=0.5,
#     #                                                    max_stretch=2)
#     max_angle_diffs = polygon_utils.compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)

#     return max_angle_diffs


# class ContourEval:
#     def __init__(self, coco_gt, coco_dt):
#         """
#         @param coco_gt: coco object with ground truth annotations
#         @param coco_dt: coco object with detection results
#         """
#         self.coco_gt = coco_gt  # ground truth COCO API
#         self.coco_dt = coco_dt  # detections COCO API

#         self.img_ids = sorted(coco_gt.getImgIds())
#         self.cat_ids = sorted(coco_dt.getCatIds())

#     def evaluate(self, pool=None):
#         gts = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=self.img_ids))
#         dts = self.coco_dt.loadAnns(self.coco_dt.getAnnIds(imgIds=self.img_ids))

#         _gts = defaultdict(list)  # gt for evaluation
#         _dts = defaultdict(list)  # dt for evaluation
#         for gt in gts:
#             _gts[gt['image_id'], gt['category_id']].append(gt)
#         for dt in dts:
#             _dts[dt['image_id'], dt['category_id']].append(dt)
#         evalImgs = defaultdict(list)  # per-image per-category evaluation results

#         # Compute metric
#         args_list = []
#         # i = 1000
#         for img_id in self.img_ids:
#             for cat_id in self.cat_ids:
#                 gts = _gts[img_id, cat_id]
#                 dts = _dts[img_id, cat_id]
#                 args_list.append((gts, dts))
#                 # i -= 1
#             # if i <= 0:
#             #     break
#         if pool is None:
#             measures_list = []
#             for args in tqdm(args_list, desc="Contour metrics"):
#                 measures_list.append(compute_contour_metrics(args))
#         else:
#             measures_list = list(tqdm(pool.imap(compute_contour_metrics, args_list), desc="Contour metrics", total=len(args_list)))
#         measures_list = [measure for measures in measures_list for measure in measures]  # Flatten list
#         # half_tangent_cosine_similarities_list, edge_distances_list = zip(*measures_list)
#         # half_tangent_cosine_similarities_list = [item for item in half_tangent_cosine_similarities_list if item is not None]
#         measures_list = [value for value in measures_list if value is not None]
#         max_angle_diffs = np.array(measures_list)
#         max_angle_diffs = max_angle_diffs * 180 / np.pi  # Convert to degrees

#         return max_angle_diffs







    '''
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))
            
    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # convert to coco bbox format: xmin, ymin, w, h
            boxes = prediction["boxes"]
            x1, y1, x2, y2 = boxes.unbind(1)
            boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
            
            boxes = boxes.tolist()
            
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            labels = [self.ann_labels[l] for l in labels]

            coco_results.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results
    
    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            labels = [self.ann_labels[l] for l in labels]

            rles = [
                mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results
    '''
    
