# R-PolyGCN

A PyTorch implementation of R-PolyGCN [1] and using simple Mask R-CNN repository.

The original repository of R-PolyGCN is https://github.com/Miracle2333/R-PolyGCN but seems it is not working.

The code is based largely on [Pytorch Simple Mask R-CNN](https://github.com/Okery/PyTorch-Simple-MaskRCNN).

[1] Zhao, K., Kamran, M., & Sohn, G. (2020). Boundary Regularized Building Footprint Extraction From Satellite Images Using Deep Neural Network. arXiv preprint arXiv:2006.13176.

## Requirements

- **Windows** or **Linux**, with **Python ≥ 3.6**

- **[PyTorch](https://pytorch.org/) ≥ 1.4.0**

- **matplotlib**, **OpenCV** - visualizing images and results

- **[pycocotools](https://github.com/cocodataset/cocoapi)** - for COCO dataset and evaluation


You can also use conda and environment.yml file to recreate the environment:
```
conda env create -f environment.yml 
```

I added Recal@0.5 to the source code of pycocotools and installed it. 
Below, you can find how I managed to do it:
1) Clone the pycocotools repository
```
git clone https://github.com/ppwwyyxx/cocoapi.git
```
2) Install it:
```
pip install cython
cd cocoapi/PythonAPI
python setup.py develop  # this way, changes will apply without any rebuild
```
3) Change 3 lines of pycocotools/cocoeval.py so that it reports IOUs at 0.5 and 0.75 too:
```
stats = np.zeros((14,))  # line 459
# Add 2 below lines after line 471
stats[12] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
stats[13] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2]) 
```

## Datasets

This repository supports SpaceNet dataset in COCO format.

In order to use SpaceNet dataset you have to:
1) download raw SpaceNet2 datasets from AWS (eg Vegas scene).
2) convert it to COCO format and split it randomly using spacenet2coco.ipynb

If you want to train your own dataset, you may:

- write the correponding dataset code

- convert your dataset to COCO-style

**MS COCO 2017**: ```http://cocodataset.org/```

COCO dataset directory should be like this:
```
coco2017/
    annotations/
        instances_train2017.json
        instances_val2017.json
        ...
    train2017/
        000000000009.jpg
        ...
    val2017/
        000000000139.jpg
        ...
```

The code will check the dataset first before start, filtering samples without annotations.

## Training
You can run the training using the below command example, obviously you must set the arguments according to yours. 
```
python train.py --use-cuda --iters -1 --dataset coco --data-dir /data/Vegas_coco_random_splits
```
or modify the parameters in ```run.sh```, and run:
```
bash ./run.sh
```

Note: This is a simple model and only supports ```batch_size = 1```. 

The code will save and resume automatically using the checkpoint file.

## Evaluation

Run the below command to evaluate the model using your preferred checkpoint.
```
python eval.py --ckpt-path maskrcnn_coco-35.pth
```

## Demo
Run the below command for your preferred checkpoint.
```
python demo.py --ckpt-path maskrcnn_coco-35.pth
```

![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)

You can also see the training/validation logs using tensorboard:
```
tensorboard --logdir maskrcnn_results/logs/train
```