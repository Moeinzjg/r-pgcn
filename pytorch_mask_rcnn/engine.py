import sys
import time

import torch

from .utils import Meter, TextArea, iou_from_poly, mask_from_poly
try:
    from .datasets import CocoEvaluator, prepare_for_coco, prepare_for_coco_polygon
except:
    pass


def train_one_epoch(model, trainable, optimizer, data_loader, device, epoch, args, writer):

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i

        image = image.to(device)
        target.pop('global_polygons')
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()

        losses = model(image, target)

        if trainable == 'MASK':
            losses['roi_polygon_loss'] = torch.tensor(0, requires_grad=False)
            losses['roi_edge_loss'] = torch.tensor(0, requires_grad=False)
            losses['roi_vertex_loss'] = torch.tensor(0, requires_grad=False)

        elif trainable == 'FAGCN':
            losses['roi_classifier_loss'] = torch.tensor(0, requires_grad=False)
            losses['roi_box_loss'] = torch.tensor(0, requires_grad=False)
            losses['roi_mask_loss'] = torch.tensor(0, requires_grad=False)

            losses['rpn_objectness_loss'] = torch.tensor(0, requires_grad=False)
            losses['rpn_box_loss'] = torch.tensor(0, requires_grad=False)

        total_loss = sum(losses.values())
        m_m.update(time.time() - S)

        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)

        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

            # log into tensorboard
            for k in losses.keys():
                writer.add_scalar(k, losses[k], num_iters)
            writer.add_scalar('total_loss', total_loss, num_iters)

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters


def evaluate(model, data_loader, device, args, generate=True, poly=False):
    iter_eval = None
    poly_iou = None
    if generate:
        iter_eval, poly_iou = generate_results(model, data_loader, device, args, poly=True)

    dataset = data_loader
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)
    coco_evaluator_rpolygcn = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")
    rpolygcn_results = torch.load(args.rpolygcn_results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    coco_evaluator_rpolygcn.accumulate(rpolygcn_results)

    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
    # if poly:
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator_rpolygcn.summarize()

    output_rpolygcn = sys.stdout
    sys.stdout = temp

    return output, output_rpolygcn, iter_eval, poly_iou


# generate results file
@torch.no_grad()
def generate_results(model, data_loader, device, args, poly=False):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    coco_results_poly = []
    if poly:
        poly_iou = []
    else:
        poly_iou = None
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v for k, v in target.items()}

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image, target)
        m_m.update(time.time() - S)

        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        prediction[target["image_id"].item()].update({'masks_from_polygons': mask_from_poly(output['polygons'].cpu(),
                                                                                            image.shape[2],
                                                                                            image.shape[1])})

        # coco evaluation on masks
        coco_results.extend(prepare_for_coco(prediction))
        # on masks from polygons
        coco_results_poly.extend(prepare_for_coco_polygon(prediction))

        # evaluation on polygons
        if poly:
            pred = output['polygons']
            gt = target['global_polygons']
            iou, _ = iou_from_poly(pred, gt, image.shape[2], image.shape[1])
            poly_iou.append(iou)

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
    if poly:
        torch.save(coco_results_poly, args.rpolygcn_results)

    return A / iters, poly_iou

