import os
from re import X
from PIL import Image

import torch
import numpy as np

from .generalized_dataset import GeneralizedDataset


class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False, num_points=16):
        super().__init__()
        from pycocotools.coco import COCO

        self.data_dir = data_dir
        self.split = split
        self.train = train
        self.num_points = num_points

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

    def make_polygon(self, poly, shape, bbox):  # TODO: check if x and y are right not y, x
        EPS = 1e-7
        poly_temp = poly.copy()
        poly_temp = np.array(poly_temp).reshape(-1, 2)  # [x, y]
        poly_temp[:, 0] = np.clip(poly_temp[:, 0], 0 + EPS, shape[0] - EPS)
        poly_temp[:, 1] = np.clip(poly_temp[:, 1], 0 + EPS, shape[1] - EPS)
        poly_temp = np.floor(poly_temp).astype(np.int32)

        polygon = self.uniform_sample(poly_temp, self.num_points)
        arr_polygon = np.ones((self.num_points, 2), np.float32) * 0.
        arr_polygon[:, :] = polygon

        # convert coordinates from global to local i.e. [0, 1]
        x_min, y_min, w, h = bbox.T

        x_max = x_min + w
        y_max = y_min + h

        x_min = max(0, x_min)
        x_max = min(shape[1] - 1, x_max)

        y_center = y_min + (1 + h) / 2. # Bounding box finishing Layer

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(shape[0] - 1, y_max)

        xs = arr_polygon[:, 0]
        ys = arr_polygon[:, 1]
        
        xs = (xs - x_min) / float(patch_w)
        ys = (ys - (y_min - top_margin)) / float(patch_w)

        xs = np.clip(xs, 0 + EPS, 1 - EPS) # between epsilon and 1-epsilon
        ys = np.clip(ys, 0 + EPS, 1 - EPS)

        arr_polygon[:, 0] = xs
        arr_polygon[:, 1] = ys

        return arr_polygon

    def uniform_sample(self, pgtnp_px2, newpnum):

        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum  # Hypothesis Enhanced Features
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths

        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        polygons = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
                poly = ann['segmentation'][0]

                polygon = self.make_polygon(poly, mask.shape)
                polygons.append(polygon)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            polygons = torch.tensor(polygons, dtype=torch.float32)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks, polygons=polygons)
        return target
