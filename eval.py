import argparse
import os
import time
import re
import math

import torch

import pytorch_mask_rcnn as pmr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda:
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    d_test = pmr.datasets(args.dataset, args.data_dir, "test", train=True)  # COCO 2017

    print(args)
    num_classes = max(d_test.classes) + 1

    if 'fpn' in args.backbone:
        backbone_name = re.findall('(.*?)_fpn', args.backbone)[0]
        model = pmr.maskrcnn_resnet_fpn(pretrained=False, num_classes=num_classes,
                                        pretrained_backbone=True, backbone_name=backbone_name).to(device)
    else:
        model = pmr.maskrcnn_resnet50(False, num_classes, pretrained_backbone=True).to(device)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    # print(checkpoint["eval_info"])
    del checkpoint

    if cuda:
        torch.cuda.empty_cache()

    print("\nevaluating...\n")

    B = time.time()
    eval_output, rpolygcn_eval_output, iter_eval, poly_iou, poly_maxtan = pmr.evaluate(model, d_test, device, args)
    B = time.time() - B
    for bf in eval_output.buffer:
        print(bf)
    print(eval_output.get_AP())
    print('---------------------------- R-PolyGCN ----------------------------')
    for bf in rpolygcn_eval_output.buffer:
        print(bf)
    print(rpolygcn_eval_output.get_AP())
    if iter_eval is not None:
        print("\nTotal time of this evaluation: {:.1f} s, speed: {:.1f} imgs/s".format(B, 1 / iter_eval))
    if poly_iou is not None:
        print("\n Average IOU of polygons is: {:.2f}".format(sum(poly_iou) / len(poly_iou)))
    if poly_maxtan is not None:
        print("\n Average MaxTangent of polygons is: {:.2f}".format(sum(poly_maxtan) / len(poly_maxtan) * 180/math.pi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir", default="../Vegas_coco_random_splits")
    parser.add_argument("--ckpt-path", default="maskrcnn_coco-25.pth")
    parser.add_argument("--iters", type=int, default=-1)  # number of iterations, minus means the entire dataset
    parser.add_argument("--backbone", type=str, default="resnet50_fpn", choices=["resnet50", "resnet50_fpn", "resnet101_fpn"])
    args = parser.parse_args()  # [] is needed if you're using Jupyter Notebook.

    args.use_cuda = True
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    args.rpolygcn_results = os.path.join(os.path.dirname(args.ckpt_path), "rpolygcn_results.pth")

    main(args)

