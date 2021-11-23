import os
from PIL import Image

import torch
import cv2
import numpy as np

from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(split))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(boxes):  # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1)  # new_box format: (xmin, ymin, xmax, ymax)
    
    @staticmethod
    def make_edge_mask(poly, shape):
        EPS = 1e-7
        edge_mask = np.zeros((shape[0], shape[1]))
        poly_temp = poly.copy()
        poly_temp = np.array(poly_temp).reshape(-1, 2)  # [x, y]
        poly_temp[:, 0] = np.clip(poly_temp[:, 0], 0 + EPS, shape[0] - EPS)
        poly_temp[:, 1] = np.clip(poly_temp[:, 1], 0 + EPS, shape[1] - EPS)
        poly_temp = poly_temp.reshape((-1, 1, 2))
        poly_temp = np.floor(poly_temp).astype(np.int32)
        # poly_temp = cv2.approxPolyDP(poly_temp, 0, False)[:, 0, :]
        cv2.polylines(edge_mask, [poly_temp], True, [1])  # TODO: visualize edge mask
        return edge_mask
    
    @staticmethod
    def make_vertex_mask(poly, shape):
        EPS = 1e-7
        vertex_mask = np.zeros((shape[0], shape[1]))
        poly_temp = poly.copy()
        poly_temp = np.array(poly_temp).reshape(-1, 2)  # [x, y]
        poly_temp[:, 0] = np.clip(poly_temp[:, 0], 0 + EPS, shape[0] - EPS)
        poly_temp[:, 1] = np.clip(poly_temp[:, 1], 0 + EPS, shape[1] - EPS)
        poly_temp = np.floor(poly_temp).astype(np.int32)
        # poly_temp = cv2.approxPolyDP(poly_temp, 0, False)[:, 0, :]
        vertex_mask[poly_temp[:, 0], poly_temp[:, 1]] = 1.0
        return vertex_mask

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        polys = []
        masks = []
        edge_masks = []
        vertex_masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
                poly = ann['segmentation'][0]
                polys.append(poly)

                edge_mask = self.make_edge_mask(poly, mask.shape)
                edge_mask = torch.tensor(edge_mask, dtype=torch.uint8)
                edge_masks.append(edge_mask)

                vertex_mask = self.make_vertex_mask(poly, mask.shape)
                vertex_mask = torch.tensor(vertex_mask, dtype=torch.uint8)
                vertex_masks.append(vertex_mask)


            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            edge_masks = torch.stack(edge_masks)
            vertex_masks = torch.stack(vertex_masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks, edges=edge_masks, vertices=vertex_masks)
        return target
    
    