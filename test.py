# Segmentations Production
# Author: Griffin Bishop
# Heavily adapted and changed by: Guru Deep Singh

from __future__ import print_function
from __future__ import division

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import scipy.io as sc

from config import Config
import util
from model import WNet
from evaluation_dataset import EvaluationDataset


def main():
    print("PyTorch Version: ", torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()

    ###################################
    # Image loading and preprocessing #
    ###################################
    type = "500" #BSDS data set you want to use

    #######################################
    # Generating the npy for ground truth #
    #######################################

    def BSDgt_to_npy():
        test_path = config.data_dir + "/test/segmentations_" + type
        destination = config.data_dir + "/test/segmentations_npy_" + type

        for file in os.listdir(test_path):
            if file.endswith(".mat"):
                ppath_to_file = os.path.join(test_path, file)
                mat = sc.loadmat(ppath_to_file)["groundTruth"][0, 0][0][0][0]
                mat = mat.astype('int16')

                image_name = str(file).split(".")[0]
                path = os.path.join(destination, image_name)
                np.save(path, mat)

    BSDgt_to_npy()

    evaluation_dataset = EvaluationDataset("test", type)

    evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset,
                                                        batch_size=config.test_batch_size, num_workers=4, shuffle=False)

    ###################################
    #          Model Setup            #
    ###################################

    # We will only use .forward_encoder()
    if torch.cuda.is_available():
        autoencoder = torch.load("./models/guru")
    else:
        autoencoder = torch.load("./models/guru", map_location=torch.device('cpu'))
    util.enumerate_params([autoencoder])

    ###################################
    #          Testing Loop           #
    ###################################

    autoencoder.eval()

    def combine_patches(image, patches):
        w, h = image[0].shape
        segmentation = torch.zeros(w, h)
        x, y = (0, 0)  # Start of next patch
        for patch in patches:
            if y + size > h:
                y = 0
                x += size
            segmentation[x:x + size, y:y + size] = patch
            y += size
        return segmentation

    # Because this model is unsupervised, our predicted segment labels do not
    # correspond to the actual segment labels.
    # We need to figure out what the best mapping is.
    # To do this, we will just count, for each of our predicted labels,
    # The number of pixels in each class of actual labels, and take the max in that image
    def count_predicted_pixels(predicted, actual):
        pixel_count = torch.zeros(config.k, config.k)
        for k in range(config.k):
            mask = (predicted == k)
            masked_actual = actual[mask]
            for i in range(config.k):
                pixel_count[k][i] += torch.sum(masked_actual == i)
        return pixel_count

    # Converts the predicted segmentation, based on the pixel counts
    def convert_prediction(pixel_count, predicted):
        map = torch.argmax(pixel_count, dim=1)
        for x in range(predicted.shape[0]):
            for y in range(predicted.shape[1]):
                predicted[x, y] = map[predicted[x, y]]
        return predicted

    def compute_iou(predicted, actual):
        intersection = 0
        union = 0
        for k in range(config.k):
            a = (predicted == k).int()
            b = (actual == k).int()
            # if torch.sum(a) < 100:
            #    continue # Don't count if the channel doesn't cover enough
            intersection += torch.sum(torch.mul(a, b))
            union += torch.sum(((a + b) > 0).int())
        return intersection.float() / union.float()

    def pixel_accuracy(predicted, actual):
        return torch.mean((predicted == actual).float())

    iou_sum = 0
    pixel_accuracy_sum = 0
    n = 0
    # Currently, we produce the most generous prediction looking at a single image
    for i, [images, segmentations, image_path] in enumerate(evaluation_dataloader, 0):
        #print(image_path)
        size = config.input_size
        # Assuming batch size of 1 right now
        image = images[0]
        target_segmentation = segmentations[0]

        # NOTE: We cut the images down to a multiple of the patch size
        cut_w = (image[0].shape[0] // size) * size
        cut_h = (image[0].shape[1] // size) * size

        image = image[:, 0:cut_w, 0:cut_h]

        target_segmentation = target_segmentation[:, 0:cut_w, 0:cut_h]

        patches = image.unfold(0, 3, 3).unfold(1, size, size).unfold(2, size, size)
        patch_batch = patches.reshape(-1, 3, size, size)

        if torch.cuda.is_available():
            patch_batch = patch_batch.cuda()

        seg_batch = autoencoder.forward_encoder(patch_batch)
        seg_batch = torch.argmax(seg_batch, axis=1).float()

        predicted_segmentation = combine_patches(image, seg_batch)
        prediction = predicted_segmentation.int()

        actual = target_segmentation[0].int()

        pixel_count = count_predicted_pixels(prediction, actual)
        prediction = convert_prediction(pixel_count, prediction)

        #######################################
        # Generating segmentation and saving in mat #
        #######################################
        # image_path
        image_name = str(image_path).split(".")[1].split("\\")[-1] + ".mat"
        image_path = config.data_dir + "/test/segmentations_pred_"  + type+ "/"+  image_name

        pred_object = prediction.cpu().detach().numpy()
        numpy_object = np.empty((1, 5), dtype=object)
        numpy_object[0, 0] = pred_object
        numpy_object[0, 1] = pred_object
        numpy_object[0, 2] = pred_object
        numpy_object[0, 3] = pred_object
        numpy_object[0, 4] = pred_object
        sc.savemat(image_path, mdict={'segs': numpy_object})


if __name__ == "__main__":
    main()
