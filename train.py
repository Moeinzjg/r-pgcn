import bisect
import glob
import os
import re
import time

import torch
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

import pytorch_mask_rcnn as pmr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda":
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    # ---------------------- prepare data loader ------------------------------- #

    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)

    d_val = pmr.datasets(args.dataset, args.data_dir, "val", train=True)  # set train=True for eval

    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = max(d_train.dataset.classes) + 1  # including background class
    model = pmr.maskrcnn_resnet50(False, num_classes, pretrained_backbone=True).to(device)

    if args.train_mode == "multistep":
        # Step1: Train the network till the end of localization (FA) module
        train(model, 20, d_train, d_val, args, device, trainable='MASK')

        # Step2: Only Train poly augmentor and GCN
        train(model, 30, d_train, d_val, args, device, trainable='FAGCN')

        # Step3: Train all togather
        train(model, 35, d_train, d_val, args, device, trainable='All')
    
    elif args.train_mode == "simul":
        # Train all togather
        train(model, 35, d_train, d_val, args, device, trainable='All')

    print('-------------------- Finished! --------------------')


def train(model, epochs, d_train, d_val, args, device, trainable='All'):

    args.epochs = epochs

    if trainable == 'MASK':
        args.lr_steps = [13, 17]
        for param in model.head.feature_augmentor.parameters():
            param.requires_grad = False

        for param in model.head.poly_augmentor.parameters():
            param.requires_grad = False

        for param in model.head.polygon_predictor.parameters():
            param.requires_grad = False

    elif trainable == 'FAGCN':
        args.lr_steps = [27]
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.feature_augmentor.parameters():
            param.requires_grad = True

        for param in model.head.poly_augmentor.parameters():
            param.requires_grad = True

        for param in model.head.polygon_predictor.parameters():
            param.requires_grad = True

    elif trainable == 'All':
        if args.train_mode == "simul":
            args.lr_steps = [15, 30]
        else:
            args.lr_steps = [35]
        for param in model.parameters():
            param.requires_grad = True

        for name, param in model.backbone.body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                param.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    start_epoch = 0

    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device)  # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))

    # ------------------------------- train ------------------------------------ #
    log_dir = 'maskrcnn_results'
    writer = SummaryWriter(os.path.join(log_dir, 'logs', 'train'))
    # val_writer = SummaryWriter(os.path.join(log_dir, 'logs', 'train_val'))
    
    if args.train_mode == "simul":
        scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))

        A = time.time()
        iter_train = pmr.train_one_epoch(model, trainable, optimizer, d_train, device, epoch, args, writer)
        A = time.time() - A

        trained_epoch = epoch + 1

        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path)

        if args.train_mode == "simul":
            scheduler.step()

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))

    # -------------------------------------------------------------------------- #

    print("\ntotal time of {:s} training: {:.1f} s".format(trainable, time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")

    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    parser.add_argument("--rpolygcn_results")

    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)

    parser.add_argument("--iters", type=int, default=1000, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    parser.add_argument("--train_mode", default="simul", choices=["simul", "multistep"], 
                        help="You should choose to train the network in multisteps (Mask R-CNN->FA->PolyGCN) or simultaneously")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.02 * 1 / 16  # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    if args.rpolygcn_results is None:
        args.rpolygcn_results = os.path.join(os.path.dirname(args.ckpt_path), "rpolygcn_results.pth")

    main(args)

