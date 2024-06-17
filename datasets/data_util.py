import numpy as np
import random
import torch
import cv2

def patch_shuffle(image_data, label_data, region_size=16):
    #shuffle_flag = random.uniform(0,1.)
    shuffle_flag = np.random.rand()
    if shuffle_flag > 0.5:
        return image_data, label_data
    
    c, h, w = image_data.shape
    x_interval =  h // region_size - 1
    y_interval = w // region_size - 1

    x_index1 = random.randint(0, x_interval*16)# * 16
    y_index1 = random.randint(0, y_interval*16)# * 16

    x_index2 = random.randint(0, x_interval*16) # * 16
    y_index2 = random.randint(0, y_interval*16) #* 16

    while x_index1==x_index2 and y_index1==y_index2:
        x_index2 = random.randint(0, x_interval*16)
        y_index2 = random.randint(0, y_interval*16) #*16

    #image = copy.deepcopy(image_data)
    #label = copy.deepcopy(label_data)
    image = image_data
    label = label_data

    im_patch1 = image[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size]
    im_patch2 = image[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size]

    gt_patch1 = label[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size]
    gt_patch2 = label[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size]
    
    image_data[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size] = im_patch2
    image_data[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size] = im_patch1

    label_data[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size] = gt_patch2
    label_data[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size] = gt_patch1

    image_data, label_data = random_offset(image_data, label_data, x_index1, x_index2, y_index1, y_index2, x_interval, y_interval)

    return image_data, label_data

def random_offset(image_data, label_data, x_index1, x_index2, y_index1, y_index2, x_interval, y_interval):
    h_or_v_flag = np.random.rand() #determinte which direction to conduct offset
    H_offset = h_or_v_flag > 0.5
    region_size = 16
    offset_dis = random.randint(0,16)
    if offset_dis == 0:
        return image_data, label_data

    if H_offset:
        #random offset along horizon direction
        x_idx = random.randint(0, x_interval*16) #* 16
        start_idx = random.randint(0, y_interval*16)
        end_idx = random.randint(start_idx + 16, (y_interval+1)*16) #* 16
        #start_idx = start_idx * 16

        im_patch = image_data[:, x_idx:x_idx+region_size, start_idx:end_idx]
        gt_patch = label_data[:,x_idx:x_idx+region_size, start_idx:end_idx]
        #patch_len = end_idx - start_idx

        replace_or_zero = np.random.rand()
        if replace_or_zero > 0.5: #replace
            #if replace_or_zero > 0.75:#forward 
            bf = int((replace_or_zero>0.75)*2-1)
            new_im_patch = torch.cat([im_patch[:, :, -bf*offset_dis:], im_patch[:, :, :-bf*offset_dis]], dim=2)
            new_gt_patch = torch.cat([gt_patch[:, :, -bf*offset_dis:], gt_patch[:, :, :-bf*offset_dis]], dim=2)
            #else:#backward
            #    new_im_patch = torch.cat([im_patch[:,:, offset_dis:], im_patch[:, :, :offset_dis]], dim=2)
            #    new_gt_patch = torch.cat([gt_patch[:,:, offset_dis:], gt_patch[:, :, :offset_dis]], dim=2)

            image_data[:, x_idx:x_idx+region_size, start_idx:end_idx] = new_im_patch 
            label_data[:,x_idx:x_idx+region_size, start_idx:end_idx] = new_gt_patch
        else: #random fill
            random_im_patch = torch.rand(3, 16, offset_dis) * 2 -1 # to -1---1
            random_gt_patch = torch.ones(1,16,offset_dis) * 50
            if replace_or_zero < 0.25:#forward 
                new_im_patch = torch.cat([random_im_patch, im_patch[:, :, :-offset_dis]], dim=2)
                new_gt_patch = torch.cat([random_gt_patch, gt_patch[:, :, :-offset_dis]], dim=2)
            else:#backward
                new_im_patch = torch.cat([im_patch[:,:, offset_dis:], random_im_patch], dim=2)
                new_gt_patch = torch.cat([gt_patch[:,:, offset_dis:], random_gt_patch], dim=2)

            image_data[:, x_idx:x_idx+region_size, start_idx:end_idx] = new_im_patch 
            label_data[:,x_idx:x_idx+region_size, start_idx:end_idx] = new_gt_patch
    else:
        #random offset along horizon direction
        y_idx = random.randint(0, y_interval*16) #* 16
        start_idx = random.randint(0, x_interval*16)
        end_idx = random.randint(start_idx + 16, (x_interval+1)*16)
        #start_idx = start_idx * 16

        im_patch = image_data[:, start_idx:end_idx, y_idx:y_idx+region_size]
        gt_patch = label_data[:, start_idx:end_idx, y_idx:y_idx+region_size]
        patch_len = end_idx - start_idx

        replace_or_zero = np.random.rand()
        if replace_or_zero > 0.5: #replace
            #if replace_or_zero > 0.75:#forward 
            bf = int((replace_or_zero>0.75)*2-1)
            new_im_patch = torch.cat([im_patch[:,-bf*offset_dis:, :], im_patch[:, :-bf*offset_dis,:]], dim=1)
            new_gt_patch = torch.cat([gt_patch[:,-bf*offset_dis:, :], gt_patch[:, :-bf*offset_dis,:]], dim=1)
            #else:#backward
            #    new_im_patch = torch.cat([im_patch[:,offset_dis:,:], im_patch[:, :offset_dis, :]], dim=1)
            #    new_gt_patch = torch.cat([gt_patch[:,offset_dis:,:], gt_patch[:, :offset_dis, :]], dim=1)

            image_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_im_patch 
            label_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_gt_patch
        else: #random fill
            random_im_patch = torch.rand(3, offset_dis, 16) * 2 -1
            random_gt_patch = torch.ones(1,offset_dis, 16) * 50
            if replace_or_zero < 0.25:#forward 
                new_im_patch = torch.cat([random_im_patch, im_patch[:, :-offset_dis, :]], dim=1)
                new_gt_patch = torch.cat([random_gt_patch, gt_patch[:, :-offset_dis,:]], dim=1)
            else:#backward
                new_im_patch = torch.cat([im_patch[:,offset_dis:, :], random_im_patch], dim=1)
                new_gt_patch = torch.cat([gt_patch[:,offset_dis:, :], random_gt_patch], dim=1)
            
            image_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_im_patch 
            label_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_gt_patch

    return image_data, label_data


def select_label(label_patch):
    labels = np.unique(label_patch)
    index1 = np.where(label_patch==labels[0])
    index2 = np.where(label_patch==labels[1])
    size1 = index1[0].size
    size2 = index2[0].size

    patch_label1_1 = np.zeros_like(label_patch)
    patch_label1_2 = np.zeros_like(label_patch)
    index1_1 = (index1[0][:size1//2], index1[1][:size1//2])
    index1_2 = (index1[0][size1//2:], index1[1][size1//2:])
    patch_label1_1[index1_1] = 1
    patch_label1_2[index1_2] = 1

    patch_label2_1 = np.zeros_like(label_patch)
    patch_label2_2 = np.zeros_like(label_patch)
    index2_1 = (index2[0][:size2//2], index2[1][:size2//2])
    index2_2 = (index2[0][size2//2:], index2[1][size2//2:])
    patch_label2_1[index2_1] = 1
    patch_label2_2[index2_2] = 1
    patchs = np.concatenate([patch_label1_1[None,:,:], patch_label1_2[None, :, :], patch_label2_1[None, :, :], patch_label2_2[None,:,:]], axis=0)

    return patchs


def collate_fn(data):
    im, label, dis2bd = zip(*data)
    images = torch.stack(im,0)
    labels = torch.stack(label,0)
    dis2boundary = torch.stack(dis2bd,0)

    return images, labels, dis2boundary
