import numpy as np
import flow_transforms
import torchvision.transforms as transforms
import torch
import cv2
from .BSD500 import *
from .data_util import patch_shuffle


def get_distance(label):

    height, width = label.shape[:2]
    label += 1
    categories = np.unique(label)
        
    if 0 in categories:
         raise RuntimeError('invalid category')

    label = cv2.copyMakeBorder(label, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dis2boundary = np.zeros((2, height+2, width+2), dtype=np.float32)

    for category in categories:
        img = (label == category).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)
        nearCord = place[labels-1,:]

        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        
        nearPixel = np.zeros((2, height+2, width+2))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel    
        dis2boundary[:, img > 0] = diff[:, img > 0]     

    dis2boundary = dis2boundary[:, 1:-1, 1:-1]

    return dis2boundary


def BDS500(path_imgs, input_transform, co_transform, target_transform, flag):
    path_imgs = path_imgs.strip()
    path_label = path_imgs.replace('_img.jpg', '_label.png')

    img = cv2.imread(path_imgs)[:, :, ::-1].astype(np.float32)
    gtseg = cv2.imread(path_label)[:,:,:1]

    assert np.max(gtseg) <= 50 and np.min(gtseg) >= 0
    img, gtseg = co_transform([img], gtseg)
    image = input_transform(img[0])
    label = target_transform(gtseg)

    if flag == 'train':
        image, label = patch_shuffle(image, label)
        image, label = patch_shuffle(image, label)

    gt_ = label.numpy().copy()
    gt_ = np.squeeze(gt_,0)
    dis2boundary = get_distance(gt_)
    dis2boundary = torch.tensor(dis2boundary)
    return image, label, dis2boundary


def get_transform(args, flag):
    crop_shape = (args.train_img_height, args.train_img_width)
    mean = args.dataset_mean
    
    val_crop = (208,208)
    mean1 = [0,0,0]
    std1=[255,255,255]
    std2=[1,1,1]
                        
    if flag == 'train':
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=mean1, std=std1),
            transforms.Normalize(mean=mean, std=std2)
        ])
        co_transform = flow_transforms.Compose([
                flow_transforms.RandomCrop(crop_shape),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip()])
    else:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean = mean1, std=std1),
            transforms.Normalize(mean = mean, std=std2)
        ])
        co_transform = flow_transforms.Compose([
                flow_transforms.CenterCrop(val_crop)
            ])

    return input_transform, co_transform











