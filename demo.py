import torch
import pytorch_mask_rcnn as pmr


use_cuda = True
dataset = "coco"
ckpt_path = "maskrcnn_coco-25.pth"
data_dir = "../Vegas_coco_random_splits/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(dataset, data_dir, "test", train=True)
d = torch.utils.data.DataLoader(ds, shuffle=False)

model = pmr.maskrcnn_resnet50(False, max(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

for p in model.parameters():
    p.requires_grad_(False)

# Quantitative measures for edge and vertex
for i, (image, target) in enumerate(d):
    image = image.to(device)[0]
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)
    
    prediction = {k: v.cpu() for k, v in result.items()}

    pred_edges = prediction['edges'].numpy()
    pred_vertices = prediction['vertices'].numpy()

    gt_edges = target['edges'].cpu().numpy()
    gt_vertices = target['vertices'].cpu().numpy()
    import pdb; pdb.set_trace()
    # Average Precision



# Visualization
num_images = 6

for i, (image, target) in enumerate(d):
    image = image.to(device)[0]
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)
    # TODO: add quatitative metrics for edge and vertex
    # print(result)
    pmr.show(image, result, ds.classes, "./images/output{}.jpg".format(i))

    if i >= num_images - 1:
        break
