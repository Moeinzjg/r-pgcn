import os
import argparse

import torch
import pytorch_mask_rcnn as pmr


def main(args):
    use_cuda = True
    dataset = args.dataset
    ckpt_path = args.ckpt_path
    data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if device.type == "cuda":
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    ds = pmr.datasets(dataset, data_dir, "test", train=True)
    d = torch.utils.data.DataLoader(ds, shuffle=True)

    model = pmr.maskrcnn_resnet50(False, max(ds.classes) + 1).to(device)
    model.eval()
    model.head.score_thresh = 0.5

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        del checkpoint

    for p in model.parameters():
        p.requires_grad_(False)

    # Visualization
    num_images = args.num_img

    for i, (image, target) in enumerate(d):
        image = image.to(device)[0]

        global_poly = target['global_polygons']
        target.pop('global_polygons')
        target = {k: v.to(device) for k, v in target.items()}
        target['global_polygons'] = global_poly

        with torch.no_grad():
            result = model(image)
        pmr.show(image, result, target, ds.classes, "./maskrcnn_results/images/output{}".format(i))

        if i >= num_images - 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data_dir", default="../Vegas_coco_random_splits")
    parser.add_argument("--ckpt_path", default="maskrcnn_coco-25.pth")
    parser.add_argument("--num_img", default=3, type=int)
    args = parser.parse_args()

    args.use_cuda = True
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    args.rpolygcn_results = os.path.join(os.path.dirname(args.ckpt_path), "rpolygcn_results.pth")

    main(args)
