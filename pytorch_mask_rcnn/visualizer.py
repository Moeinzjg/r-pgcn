# Copyright (c) Facebook, Inc. and its affiliates.
import colorsys
import numpy as np
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import torch

__all__ = ["ColorMode", "VisImage", "Visualizer", "show"]

_SMALL_OBJECT_AREA_THRESH = 1000


def show(images, preds=None, targets=None, classes=None, save_path=None):
    """
    Show the image, with or without the pred/target.

    args:
        images (tensor[B, 3, H, W] or List[tensor[3, H, W]]): RGB channels, value range: [0.0, 1.0]
        preds (Dict[str: tensor]): current support "boxes", "labels", "scores", "masks", "polygons"
           all tensors should be of the same length, assuming N
           boxes: shape=[N, 4], format=(xmin, ymin, xmax, ymax)
           masks: shape=[N, H, W], dtype can be one of [torch.bool, torch.uint8, torch.float]
        targets (Dict[str: tensor]): current support "boxes", "labels", "scores", "masks", "polygons"
           all tensors should be of the same length, assuming N
           boxes: shape=[N, 4], format=(xmin, ymin, xmax, ymax)
           masks: shape=[N, H, W], dtype can be one of [torch.bool, torch.uint8, torch.float]
        classes (Tuple[str] or Dict[int: str]): class names
        save (str): path where to save the figure
    """
    if isinstance(images, torch.Tensor) and images.dim() == 3:
        images = [images]
    if isinstance(preds, dict):
        preds = [preds]
    if isinstance(targets, dict):
        targets = [targets]
    if isinstance(save_path, str):
        if len(images) == 1:
            prefix, ext = os.path.splitext(save_path)
            save_path_poly = ["{}_poly{}".format(prefix, ext)]
            save_path = [save_path]
        else:
            prefix, ext = os.path.splitext(save_path)
            save_path = ["{}_{}{}".format(prefix, i + 1, ext) for i in range(len(images))]
            save_path_poly = ["{}_{}_poly{}".format(prefix, i + 1, ext) for i in range(len(images))]

    for i in range(len(images)):
        fig = Visualizer(images[i])
        # fig_poly = Visualizer(images[i])

        if preds is not None:
            if targets is not None:
                fig.draw_instance_predictions(preds[i], targets[i], classes)
            else:
                fig.draw_instance_predictions(preds[i], classes)
            # fig_poly.draw_instance_predictions(preds[i], classes)
        fig.show()
        if save_path is not None:
            fig.save_plot(save_path[i])
            # fig_poly.save_plot(save_path_poly[i])


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """


class GenericMask:
    def __init__(self, mask, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        self.mask = mask.astype("uint8") # ndarray, shape=(h, w)
        self.polygons = self.mask_to_polygons(self.mask)

    def mask_to_polygons(self, mask):
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, True

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class GenericPolygon:
    def __init__(self, polygon):
        self._polygon = None
        self.polygon = polygon


# class GenericEdge:
#     def __init__(self, edge, height, width):
#         self._edge = None
#         self.height = height
#         self.width = width

#         self.edge = edge.astype("uint8")  # ndarray, shape=(h, w)


# class GenericVertex:
#     def __init__(self, vertex, height, width):
#         self._vertex = None
#         self.height = height
#         self.width = width

#         self.vertex = vertex.astype("uint8")  # ndarray, shape=(h, w)
#         self.points = self.vertex_mask_to_points(self.vertex)

    # def vertex_mask_to_points(self, vertex):
    #     vertex = np.ascontiguousarray(vertex)  # some versions of cv2 does not support incontiguous arr
    #     ver = np.nonzero(vertex)
    #     res = np.vstack((ver[1], ver[0]))
        
    #     if res.shape[1] == 0:  # empty mask
    #         return [], False

    #     return res, True


def _create_text_labels(classes, scores, class_names):
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")  # shape (H, W, 3) (RGB) in uint8 type


class Visualizer:
    # TODO implement a fast, rasterized version using OpenCV
    def __init__(self, img_rgb, scale=1.0, instance_mode=ColorMode.IMAGE):
        img_rgb = img_rgb.cpu().permute(1, 2, 0) * 255
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self.output_poly = VisImage(self.img, scale=scale)
        self.output_gtpoly = VisImage(self.img, scale=scale)

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 35, 10 // scale
        )
        self._instance_mode = instance_mode

    def __call__(self, *args, **kwargs):
        return self.draw_instance_predictions(*args, **kwargs)

    def draw_instance_predictions(self, predictions, targets=None, class_names=None, thing_colors=None):
        boxes = predictions["boxes"] if "boxes" in predictions else None
        scores = predictions["scores"] if "scores" in predictions else None
        classes = predictions["labels"] if "labels" in predictions else None
        masks = predictions["masks"] if "masks" in predictions else None
        polygons = predictions["polygons"] if "polygons" in predictions else None

        # edges = predictions["edges"] if "edges" in predictions else None
        # vertices = predictions["vertices"] if "vertices" in predictions else None

        labels = _create_text_labels(classes.tolist(), scores, class_names)

        # target polygons
        gt_polygons = targets["global_polygons"] if "polygons" in targets else None
        gt_polygons = list(map(torch.squeeze, gt_polygons))

        if self._instance_mode == ColorMode.SEGMENTATION and thing_colors is not None:
            colors = [
                self._jitter([x / 255 for x in thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            polygons=polygons,
            gt_polygons=gt_polygons,
            # edges=edges,
            # vertices=vertices,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output, self.output_poly, self.output_gtpoly

    def overlay_instances(self, boxes=None, labels=None, masks=None,
                          polygons=None, gt_polygons=None, edges=None,
                          vertices=None, assigned_colors=None, alpha=0.5):
        num_instances = 0
        if boxes is not None:
            boxes = np.asarray(boxes.cpu())
            num_instances = len(boxes)

        if masks is not None:
            if masks.is_floating_point():
                masks = masks > 0.5
            m = np.asarray(masks.cpu())
            masks = [GenericMask(x, self.output.height, self.output.width) for x in m]
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)

        if polygons is not None:
            p = np.array(polygons.cpu())
            polygons = [GenericPolygon(x) for x in p]
            if num_instances:
                assert len(polygons) == num_instances
            else:
                num_instances = len(polygons)

        if gt_polygons is not None:
            gt_polygons = [GenericPolygon(np.array(x.cpu())) for x in gt_polygons]
            num_instances_gt = len(gt_polygons)

        # if edges is not None:
        #     if edges.is_floating_point():
        #         edges = edges > 0.5
        #     m = np.asarray(edges.cpu())
        #     edges = [GenericEdge(x, self.output.height, self.output.width) for x in m]
        #     if num_instances:
        #         assert len(edges) == num_instances
        #     else:
        #         num_instances = len(edges)

        # if vertices is not None:
        #     if vertices.is_floating_point():
        #         vertices = vertices > 0.5
        #     m = np.asarray(vertices.cpu())
        #     vertices = [GenericVertex(x, self.output.height, self.output.width) for x in m]
        #     if num_instances:
        #         assert len(vertices) == num_instances
        #     else:
        #         num_instances = len(vertices)

        if labels is not None:
            assert len(labels) == num_instances

        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]

        if num_instances == 0 and num_instances_gt == 0:
            return self.output, self.output_poly, self.output_gtpoly

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            polygons = [polygons[idx] for idx in sorted_idxs] if polygons is not None else None
            # edges = [edges[idx] for idx in sorted_idxs] if edges is not None else None
            # vertices = [vertices[idx] for idx in sorted_idxs] if vertices is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        # if edges is not None:
        #     self.all_edges = edges[0].edge.copy()

        # if vertices is not None:
        #     self.all_vertices = vertices[0].vertex.copy()

        for i in range(num_instances):
            color = assigned_colors[i]
            # if boxes is not None:
            #     self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                plygons = masks[i].polygons
                if plygons[1]:
                    for segment in plygons[0]:
                        self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha, mask=True)

            if polygons is not None:
                polygon = polygons[i].polygon
                self.draw_polygon(polygon, [1.0, 1.0, 1.0], edge_color=[0.0, 1.0, 0.0], alpha=alpha, mask=False)

            # if edges is not None:
            #     self.all_edges[edges[i].edge == 1] = 1

            # if vertices is not None:
            #     points = vertices[i].points
            #     if points[1]:
            #         self.draw_polypoint(points[0].transpose(), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y0 <= 5:
                        text_pos = (x0, y1)
                    else:
                        text_pos = (x0, y0)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.6, 2)
                    * 0.5
                    * self._default_font_size
                )
                # self.draw_text(
                #     labels[i],
                #     text_pos,
                #     color=lighter_color,
                #     horizontal_alignment=horiz_align,
                #     font_size=font_size,
                # )
        # draw gt polygons
        for i in range(num_instances_gt):
            if gt_polygons is not None:
                gt_polygon = gt_polygons[i].polygon
                self.draw_polygon(gt_polygon, [1.0, 1.0, 1.0], edge_color=[0.0, 1.0, 0.0], alpha=alpha, mask=False, gt=True)

        return self.output, self.output_poly, self.output_gtpoly

    def draw_text(self, text, position, font_size=None, 
                  color="g", horizontal_alignment="center"):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 1.0, "pad": 1.0, "edgecolor": "none"},
            verticalalignment="bottom",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            #rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5, fill=True, mask=True, gt=False):
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=fill,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1)
        )
        if mask:
            self.output.ax.add_patch(polygon)
            return self.output
        elif gt:
            self.output_gtpoly.ax.add_patch(polygon)
            vert = polygon.get_xy()
            self.output_gtpoly.ax.plot(vert[:, 0], vert[:, 1], '.', color=[1.0, 0.0, 0.0], alpha=alpha)
            return self.output_gtpoly
        else:
            self.output_poly.ax.add_patch(polygon)
            vert = polygon.get_xy()
            self.output_poly.ax.plot(vert[:, 0], vert[:, 1], '.', color=[1.0, 0.0, 0.0], alpha=alpha)
            return self.output_poly

    def draw_polypoint(self, segment, color, alpha=0.5):

        self.output.ax.plot(segment[:, 0], segment[:, 1], '*', color=color, alpha=alpha)

        return self.output

    def show(self):
        H, W = self.img.shape[:2]
        self.fig = plt.figure(figsize=(W / 72, H / 72))

        # plot mask prediction
        ax = self.fig.add_subplot(131)
        img_out = self.output.get_image()
        ax.imshow(img_out)
        ax.set_title("mask prediction")
        ax.axis("off")

        # plot polygon prediction
        ax = self.fig.add_subplot(132)
        img_out = self.output_poly.get_image()
        ax.imshow(img_out)
        ax.set_title("polygon prediction")
        ax.axis("off")

        # plot polygon gt
        ax = self.fig.add_subplot(133)
        img_out = self.output_gtpoly.get_image()
        ax.imshow(img_out)
        ax.set_title("polygon gt")
        ax.axis("off")

        # plot edge prediction
        # if False:
        #     ax2 = self.fig.add_subplot(132)
        #     img_out2 = self.img.copy()
        #     img_out2[self.all_edges == 1] = [255, 0, 0]  # add edges
        #     ax2.imshow(img_out2.astype("uint8"))
        #     ax2.set_title("edge prediction")
        #     ax2.axis("off")

        # # plot vertex prediction
        # if False:
        #     ax3 = self.fig.add_subplot(133)
        #     img_out3 = self.img.copy()
        #     img_out3[self.all_vertices == 1] = [255, 0, 0]  # add vertices
        #     ax3.imshow(img_out3.astype("uint8"))
        #     ax3.set_title("vertex prediction")
        #     ax3.axis("off")

        plt.show()

    def save_plot(self, file_path):
        self.fig.savefig(file_path, bbox_inches='tight')

    def _jitter(self, color):
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color


def random_color(rgb=False, maximum=255):
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)
