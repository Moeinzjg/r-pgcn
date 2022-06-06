import os
from re import X
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
        img_path = os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"])
        with Image.open(img_path) as image:
            image = image.convert("RGB")
        return image, img_info["file_name"]

    @staticmethod
    def convert_to_xyxy(box):  # box format: (xmin, ymin, w, h)
        x, y, w, h = box.T
        return torch.stack((x, y, x + w, y + h), dim=1)  # new_box format: (xmin, ymin, xmax, ymax)

    @staticmethod
    def poly_check(poly, shape):
        EPS = 1e-7
        poly_temp = poly.copy()
        poly_temp = np.array(poly_temp).reshape(-1, 2)  # [x, y]
        poly_temp[:, 0] = np.clip(poly_temp[:, 0], 0 + EPS, shape[0] - EPS)
        poly_temp[:, 1] = np.clip(poly_temp[:, 1], 0 + EPS, shape[1] - EPS)
        poly_temp = np.floor(poly_temp).astype(np.int32)

        pnum, cnum = poly_temp.shape

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum  # Hypothesis Enhanced Features
        pgtnext_px2 = poly_temp[idxnext_p]
        
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - poly_temp) ** 2, axis=1))
 
        return edgelen_p.sum() > 0.0

    @staticmethod
    def bbox_check(bbox):
        _, _, w, h = bbox
        if w == 0 or h == 0:
            return False
        return True

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
        cv2.polylines(edge_mask, [poly_temp], True, [1])
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
        vertex_mask[poly_temp[:, 1], poly_temp[:, 0]] = 1.0
        return vertex_mask

    def make_polygon(self, poly, shape, bbox):
        EPS = 1e-7
        poly = poly[:-2]  # remove the duplicate first point at the end
        poly_temp = poly.copy()
        poly_temp = np.array(poly_temp).reshape(-1, 2)  # [x, y]
        poly_temp[:, 0] = np.clip(poly_temp[:, 0], 0 + EPS, shape[0] - EPS)
        poly_temp[:, 1] = np.clip(poly_temp[:, 1], 0 + EPS, shape[1] - EPS)
        poly_temp = np.floor(poly_temp).astype(np.int32)
        arr_polygon = poly_temp.copy().astype(np.float32)
        # self.poly_show(arr_polygon, poly_temp)

        # convert coordinates from global to local i.e. [0, 1]
        x_min, y_min, w, h = bbox

        xs = arr_polygon[:, 0]
        ys = arr_polygon[:, 1]

        xs = (xs - x_min) / float(w)
        ys = (ys - y_min) / float(h)

        xs = np.clip(xs, 0 + EPS, 1 - EPS)  # between epsilon and 1-epsilon
        ys = np.clip(ys, 0 + EPS, 1 - EPS)

        arr_polygon[:, 0] = xs
        arr_polygon[:, 1] = ys

        return arr_polygon

    @staticmethod
    def make_global_polygon(poly, shape):
        EPS = 1e-7
        poly = poly[:-2]  # remove the duplicate first point at the end
        poly_glob = poly.copy()
        poly_glob = np.array(poly_glob).reshape(-1, 2)  # [x, y]
        poly_glob[:, 0] = np.clip(poly_glob[:, 0], 0 + EPS, shape[0] - EPS)
        poly_glob[:, 1] = np.clip(poly_glob[:, 1], 0 + EPS, shape[1] - EPS)
        poly_glob = np.floor(poly_glob).astype(np.int32)

        return poly_glob
    
    def poly_show(self, poly_sampled, poly_gt):
            import matplotlib.pyplot as plt
            # close the polygons
            poly_sampled = np.concatenate([poly_sampled, np.expand_dims(poly_sampled[0], 0)], 0)
            poly_gt = np.concatenate([poly_gt, np.expand_dims(poly_gt[0], 0)], 0)
            
            plt.plot(poly_gt[:, 0], poly_gt[:, 1], 'g')
            plt.plot(poly_gt[:, 0], poly_gt[:, 1], 'gs')

            plt.plot(poly_sampled[:, 0], poly_sampled[:, 1], 'r-.')
            plt.plot(poly_sampled[:, 0], poly_sampled[:, 1], 'r*')

            plt.show()
            return 0

    @staticmethod
    def expand_bbox(bbox, image_shape):
        x_min, y_min, w, h = bbox
        x_min -= 0.1 * w
        y_min -= 0.1 * h
        w += 0.1 * w
        h += 0.1 * h

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        w_min = min(image_shape[1], w)
        h_min = min(image_shape[0], h)

        bbox = [x_min, y_min, w, h]

        return bbox
    
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        edge_masks = []
        vertex_masks = []
        polygons = []
        global_polygons = []

        if len(anns) > 0:
            for ann in anns:
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                poly = ann['segmentation'][0]
                box = self.expand_bbox(ann['bbox'], mask.shape)

                # sanity check: if it is a real polygon in image or not
                if self.poly_check(poly, mask.shape) == False:
                    continue
                if self.bbox_check(box) == False:
                    continue
                polygon = self.make_polygon(poly, mask.shape, box)
                polygons.append(polygon)

                global_polygon = self.make_global_polygon(poly, mask.shape)
                global_polygons.append(global_polygon)
                
                edge_mask = self.make_edge_mask(poly, mask.shape)
                edge_mask = torch.tensor(edge_mask, dtype=torch.uint8)
                edge_masks.append(edge_mask)

                vertex_mask = self.make_vertex_mask(poly, mask.shape)
                vertex_mask = torch.tensor(vertex_mask, dtype=torch.uint8)
                vertex_masks.append(vertex_mask)

                boxes.append(box)
                masks.append(mask)
                labels.append(ann["category_id"])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            edge_masks = torch.stack(edge_masks)
            vertex_masks = torch.stack(vertex_masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels,
                      masks=masks, edges=edge_masks, vertices=vertex_masks,
                      polygons=polygons, global_polygons=global_polygons)

        return target
