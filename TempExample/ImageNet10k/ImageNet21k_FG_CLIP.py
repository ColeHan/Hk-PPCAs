import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
from scipy.io import loadmat
from scipy.io import savemat
import torch.utils.data as data_utils
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import gc
import clip
from os import listdir
from os.path import isfile, join

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip
from torch.utils.data import Dataset,TensorDataset, DataLoader
import xml.etree.ElementTree as ET


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def data_augmentation(n_px):
    return Compose([
        # Resize(n_px, interpolation=BICUBIC), # from _transform
        # CenterCrop(n_px), # from _transform
        transforms.RandomResizedCrop(size=n_px, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        _convert_image_to_rgb, # from _transform
        ToTensor(), # from _transform
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def features32(net, x):
    x = x.type(net.conv1.weight.dtype)
    for conv, bn in [(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)]:
        x = net.relu(bn(conv(x)))
    #print(1,x.shape)
    x = net.avgpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    #print(2,x.shape)
    #x = net.attnpool(x)
    x=F.avg_pool2d(x,x.shape[2])
    return x

def features(net, x):
    x = x.type(net.conv1.weight.dtype)
#     print(x.shape)
    for conv, bn in [(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)]:
        x = net.relu(bn(conv(x)))
#     print(1,x.shape)
    x = net.avgpool(x)
#     print(x.shape)
    x = net.layer1(x)
#     print(x.shape)
    x = net.layer2(x)
#     print(x.shape)
    x = net.layer3(x)
#     print(x.shape)
    x = net.layer4(x)
#     print(2,x.shape)
    x = net.attnpool(x) # 288
#     x = x.flatten(start_dim=1)
    return x.detach().clone()


# def ParseAnoXml(path):
#     files = [f for f in listdir(path) if isfile(join(path, f))]
#     classes={}
#     nf=len(files)
#     for i in range(nf):
#         f=files[i]
#         tree = ET.parse(path+f)
#         root = tree.getroot()
#         child=root.findall("./object/name")[0]
#         fj=f[0:-4]+'.jpeg'
#         if child.text in classes.keys():
#             classes[child.text].append(fj)
#         else:
#             classes[child.text]=[fj]
#         if i%1000==0:
#             print(i,fj)
#     return classes


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # nc = 10450

    nx = 288  # 256
    # nx = 144  # 256
    # k = 20

    # train_folder='H:/LargeDataFile/ILSVRC/Data/CLS-LOC/train/'
    train_folder='K:/LargeDataset/imagenet21k_resized/imagenet21k_train/'
    names=listdir(train_folder)

    filename_cls_map = {}
    cls_filename_map = {}
    for cls, name in enumerate(names):
        filename_cls_map[name] = cls
        cls_filename_map[cls] = name


    nc = len(names)
    print(nc)
    # path = 'H:/LargeDataFile/ILSVRC/Annotations/CLS-LOC/val/'
    # classes = ParseAnoXml(path)
    # torch.save(classes, 'H:/LargeDataFile/ILSVRC/classes_val.pth')
    # classes = torch.load('H:/LargeDataFile/ILSVRC/classes_val.pth')

    # num_augmented = 10
    # print(clip.available_models())
    model, preprocess = clip.load('RN50x4', device)
    train = 0
    for j in range(8806, nc):
        if train:
            path = 'K:/LargeDataset/imagenet21k_resized/imagenet21k_train/' + names[j]
            # path = 'H:/LargeDataFile/ILSVRC/Data/CLS-LOC/train/' + names[j]
            # pathAno = 'H:/LargeDataFile/ILSVRC/Data/MINI-ImageNet/train/' + names[j]
            files = [f for f in listdir(path) if isfile(join(path, f))]
        else:
            # path='C:/Datasets/ILSVRC2016/ILSVRC/Data/CLS-LOC/val'
            # path = 'H:/LargeDataFile/ILSVRC/Data/ImageNet100/val'
            path = 'K:/LargeDataset/imagenet21k_resized/imagenet21k_val/' + names[j]
            # path = 'H:/LargeDataFile/ILSVRC/Data/CLS-LOC/val/' + names[j]
            # pathAno='C:/Datasets/ILSVRC2016/ILSVRC/Annotations/CLS-LOC/val'
            # pathAno = 'H:/LargeDataFile/ILSVRC/Data/ImageNet100/val'
            # pathAno = 'H:/LargeDataFile/ILSVRC/Data/ImageNet100/val/' + names[j]
            # files = classes[names[j]]
            files = [f for f in listdir(path) if isfile(join(path, f))]


        n_files = len(files)
        n_imgs = n_files # * num_augmented
        x_noaug = torch.zeros([n_imgs, 3, nx, nx], dtype=torch.float32)
        for i in range(n_files):
            im = Image.open(join(path, files[i]))
            x_noaug[i,:,:,:]=_transform(nx)(im)
            # for aug_i in range(num_augmented):
            #     x_aug[i * num_augmented + aug_i, :, :, :] = data_augmentation(nx)(im)
        data = TensorDataset(x_noaug)
        loader = DataLoader(data, batch_size=256, shuffle=False, drop_last=False)


        i = 0
        for images in loader:
            images = images[0].to(device)
            with torch.no_grad():
                fi = features(model.visual, images)
                # fi = features32(model.visual, images)
            if i == 0:
                X = fi
            else:
                X = torch.cat((X, fi), dim=0)
            i += 1
            # break
        p = X.shape[1]
        print(X.shape)
        if train:
            # mx,L,S=pca_model(X.float().cpu(),k)
            # name='C:/Datasets/ILSVRC2016/model_%d/%s.pth'%(p,names[j])
            # name='H:/LargeDataFile/ILSVRC/model_%d_resnet/%s.pth'%(p,names[j])
            # torch.save({'mx':mx,'L':L, 'S':S},name)
            # name = 'H:/LargeDataFile/ILSVRC/model_%d_MINI-ImageNet_resnetx4/%s.mat' % (p, names[j])
            # name = 'H:/LargeDataFile/ILSVRC/model_%d_AugmentedImageNet_288x288_resnetx4/%s.mat' % (p, names[j])
            name = 'K:/LargeDataset/imagenet21k_resized/features_train_ImageNet10k_288_resnetx4_CLIP/%s.mat' % (names[j])
            savemat(name, {'feature': X.float().cpu().numpy(),'label':filename_cls_map[names[j]]})
            #         savemat(name,{'mx':mx.cpu().numpy(),'L':L.cpu().numpy(), 'S':S.cpu().numpy()}

        else:
            # name='C:/Datasets/ILSVRC2016/features_val_641/%s.mat'%(names[j])
            # name = 'H:/LargeDataFile/ILSVRC/features_val_641_MINI-ImageNet_resnetx4/%s.mat' % (names[j])
            # name = 'H:/LargeDataFile/ILSVRC/features_val_641_AugmentedImageNet_288x288_resnetx4/%s.mat' % (names[j])
            name = 'K:/LargeDataset/imagenet21k_resized/features_val_ImageNet10k_288_resnetx4_CLIP/%s.mat' % (names[j])
            savemat(name, {'feature': X.cpu().float().numpy(),'label':filename_cls_map[names[j]]})
        print(j, name)