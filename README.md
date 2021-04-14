# Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation
##### Author 1 - Guru Deep Singh (g.d.singh@student.tudelft.nl)   
##### Author 2 - Nadine Duursman

## Inroduction
In this repository we will describe our implementation to reproduce the deep learning model: “W-Net: A Deep Model for Fully Unsupervised Image Segmentation” [[1]](https://arxiv.org/abs/1711.08506) in Pytorch. We are doing this for a student assignment for the course Deep Learning 2020 – 2021 at Delft University of Technology. 
W-Net is a deep learning model that is used for unsupervised image segmentation. This is becoming increasingly important because image labelling is time consuming and it is difficult to obtain in novel domains. The W-Net architecture consists of an encoder that outputs the image segmentations, and a decoder that reconstructs the images from these segmentations. We have modified and concatenated three existing Github repositories to do so. 

![alt text](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/Preds/pred_300/Figure_6.png)
## Adapted Repositories
- [Base Repository](https://github.com/gr-b/W-Net-Pytorch)
- [N_Cut Loss Adaptation](https://github.com/fkodom/wnet-unsupervised-image-segmentation)
- [Metrics Repository](https://github.com/KuangHaofei/BSD500-Segmentation-Evaluator)

## Repository Contents
This repository is self-contained.
##### Python Scripts
- Train a W-net model with parameters mentioned in another config file. [[Train]](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/train.py) [[Config]](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/config.py)
- Plot the graph for reconstruction and N-cut loss. [[Plot-Graph]](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/plot_loss.py)
- Create Segmentations for the test data. [[Test]](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/test.py)
- Calculate SC, PRI and VI for the segmentation. [[Metric]](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/BSD500-Segmentation-Evaluator-master/python/test_bench.py)
Some other python scripts to make the approach more modular.

##### Datasets
- [PASCAL VOC2012](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/tree/main/datasets/BSDS500val/train/images)
- [BSDS300](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/tree/main/datasets/BSDS500val/test/images_300)
- [BSDS500](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/tree/main/datasets/BSDS500val/test/images_500)

The Repository also provides a [pre-trained model](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/tree/main/models) which we trained ourselves from scratch and created [segmentations of BSDS300](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/tree/main/datasets/BSDS500val/test/segmentations_pred_300) and [segmentations of BSDS500](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/tree/main/datasets/BSDS500val/test/segmentations_pred_500).

## How to use this repository
- Clone the Repository
##### Training
- Setup the [Config file](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/config.py) (If you want to try something else)
- Execute [train.py](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/train.py)

##### Generating the Segmentations
- In [test.py](https://github.com/Guru-Deep-Singh/Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation/blob/main/test.py) select the type of dataset you want to create segmentations for "500" or "300". All the images are already placed in their respective paths. 
>*datasets/BSDS500Val/test/images_300*
> *datasets/BSDS500Val/test/images_500*

- Run the script.
