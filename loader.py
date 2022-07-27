import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageEnhance
import random

class Cityscapes(Dataset):
    def __init__(self, datasetpath, labelsetpath, kind='train'):
        self.imgs = []
        self.labels = []

        with open(datasetpath, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = line.strip('\n')
                line = line.rstrip()
                img_path = line

                if kind=="train":
                    label_path = labelsetpath + kind + line[42:-15] + 'gtFine_labelIds.png'
                elif kind=="val":
                    label_path = labelsetpath + kind + line[40:-15] + 'gtFine_labelIds.png'


                self.imgs.append(img_path)
                self.labels.append(label_path)

    def __len__(self):#30609
        length = len(self.imgs)
        return length

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])

        label = Image.open(self.labels[index])

        img_sz = 1024
        image = image.resize((img_sz, img_sz//2))
        label = label.resize((img_sz, img_sz//2))


        image = np.array(image, dtype=np.uint8)

        label = np.array(label)

        image_t = np.zeros((3, img_sz//2, img_sz))

        image_t[0, :, :] = image[:, :, 0]
        image_t[1, :, :] = image[:, :, 1]
        image_t[2, :, :] = image[:, :, 2]
        image_t = (image_t / 255) * 2 - 1

        nature = np.zeros((label.shape))
        flat = np.zeros((label.shape))
        human_vehicle = np.zeros((label.shape))

        nature[(label <= 6) | ((label >= 11) & (label <= 23))] = 1
        flat[(label >= 7) & (label <= 10)] = 1
        human_vehicle[label >= 24] = 1

        return image_t, nature, flat, human_vehicle, label


class Cityscapes_2w(Dataset):
    def __init__(self, datasetpath):
        self.imgs = []

        with open(datasetpath, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = line.strip('\n')
                line = line.rstrip()

                self.imgs.append(line)

    def __len__(self):#30609
        return length

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])

        img_sz = 1024
        image = image.resize((img_sz, img_sz//2))

        image = np.array(image, dtype=np.uint8)

        if len(image.shape)<3:
            image = image.reshape(img_sz//2, img_sz, 1)
            image = np.repeat(image, 3, axis=2)

        image_t = np.zeros((3, img_sz//2, img_sz))

        image_t[0, :, :] = image[:, :, 0]
        image_t[1, :, :] = image[:, :, 1]
        image_t[2, :, :] = image[:, :, 2]


        image_t = ((image_t -np.min(image_t))/(np.max(image_t)-np.min(image_t))) * 2 - 1


        return image_t


def prepare_data_path_txt():
    with open('../../Dataset/Cityscapes/dataset-train.txt', 'w', encoding='utf-8') as f:
        datasetpath="../../Dataset/Cityscapes/leftImg8bit/train_extra/"

        dirs=os.listdir(datasetpath)
        for dir in dirs:
            class_path = datasetpath+dir
            image_path = os.listdir(class_path)
            for img_path in image_path:
                path = class_path + "/" + img_path

                image = Image.open(path)
                image = np.array(image, dtype=np.uint8)
                if len(image.shape)<3:
                    print(path)
                    continue

                f.write(path)
                f.write('\n')

        datasetpath = "../../Dataset/Cityscapes/leftImg8bit/train/"
        dirs = os.listdir(datasetpath)
        for dir in dirs:
            class_path = datasetpath + dir
            image_path = os.listdir(class_path)
            for img_path in image_path:
                path = class_path + "/" + img_path
                image = Image.open(path)
                image = np.array(image, dtype=np.uint8)
                if len(image.shape) < 3:
                    print(path)
                    continue

                f.write(path)
                f.write('\n')

def prepare_data_label_txt():
    with open('../../Dataset/Cityscapes/dataset-val-label.txt', 'w', encoding='utf-8') as f:
        datasetpath = "../../Dataset/Cityscapes/leftImg8bit/val/"
        dirs = os.listdir(datasetpath)
        for dir in dirs:
            class_path = datasetpath + dir
            image_path = os.listdir(class_path)
            for img_path in image_path:
                path = class_path + "/" + img_path


                f.write(path)
                f.write('\n')

def prepare_openimage_txt():
    with open('../dataset/OpenImage/validation.txt', 'w', encoding='utf-8') as f:
        datasetpath = "../dataset/OpenImage/validation"
        dirs = os.listdir(datasetpath)
        for dir in dirs:
            class_path = datasetpath + "/" + dir

            f.write(class_path)
            f.write('\n')

class OpenImage(Dataset):
    def __init__(self, datasetpath):
        self.imgs = []

        with open(datasetpath, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = line.strip('\n')
                line = line.rstrip()

                self.imgs.append(line)

    def __len__(self):
        len(self.imgs)
        return length

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])

        img_sz = 512
        image = image.resize((img_sz, img_sz))

        image = np.array(image, dtype=np.uint8)

        while (len(image.shape)!=3):
            index = index+1
            image = Image.open(self.imgs[index])
            img_sz = 512
            image = image.resize((img_sz, img_sz))

            image = np.array(image, dtype=np.uint8)
        while (image.shape[2]!=3):
            index = index+1
            image = Image.open(self.imgs[index])
            img_sz = 512
            image = image.resize((img_sz, img_sz))

            image = np.array(image, dtype=np.uint8)

        image_t = ((image -np.min(image))/(np.max(image)-np.min(image))) * 2 - 1
        return image_t


def read_all_image(path, length):
    imgs_path = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()

            imgs_path.append(line)

    img_sz = 512
    image_np = np.zeros((length, img_sz, img_sz, 3))
    for index in range(length):
        image = Image.open(imgs_path[index])

        image = image.resize((img_sz, img_sz))

        image = np.array(image, dtype=np.uint8)

        if len(image.shape) < 3:
            image = image.reshape(img_sz, img_sz, 1)
            image = np.repeat(image, 3, axis=2)

        image_t = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 2 - 1
        print(index, image_t.shape)
        image_np[index] = image_t[:, :, :3]

    return image_np

class Loader_all_data(Dataset):
    def __init__(self, image_np):
        self.imgs = image_np

    def __len__(self):#30609
        length = self.imgs.shape[0]
        return length

    def __getitem__(self, index):
        image = self.imgs[index]

        return image