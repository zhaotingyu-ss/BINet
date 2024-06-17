import torch.utils.data as data
import flow_transforms
from .dataset_loader import *
import os

def get_files_list(raw_dir):
    files_list = []
    for filepath, dirnames, filenames in os.walk(raw_dir):
        for filename in filenames:
            if 'img' in filename:
                files_list.append(filepath + '/' + filename)
    return files_list

def read_voc_images(voc_dir,flag):
    fpath = os.path.join(voc_dir,'ImageSets','Segmentation',f'{flag}.txt')
    with open(fpath,'r') as f:
        images_name = f.read().split()
    print('images_name.len = ',len(images_name))
    images_path_list = []

    for image_name in images_name:
        image_path = os.path.join(voc_dir,f'{flag}',f'{image_name}_img.jpg')
        images_path_list.append(image_path)

    return images_path_list


class ListDataset(data.Dataset):
    def __init__(self, args, flag):
        self.data_root = args.data
        self.target_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
        ])
        self.debug_num = 20 
        self.debug = args.debug
        self.data_length = 0
        self.flag = flag
        self.dataset = args.dataset

        raw_dir = os.path.join(self.data_root, self.flag)
        self.img_list = get_files_list(raw_dir)
        self.BDS_ttrans,self.BDS_coTrans = get_transform(args, flag=flag)
        #if flag == 'train':
        #    random.shuffle(self.img_list)

        if self.debug:
            self.img_list = self.img_list[:self.debug_num]
        self.BDS_length = len(self.img_list)
        if len(self.img_list) > self.data_length:
            self.data_length = len(self.img_list)

    def __getitem__(self, index):
        data = []
        im_path = self.img_list[index]
        bds_im, bds_label, distance = BDS500(im_path, self.BDS_ttrans, self.BDS_coTrans, self.target_transform, self.flag)

        data.append(bds_im)
        data.append(bds_label)
        data.append(distance)

        return data

    def __len__(self):
        return self.data_length


