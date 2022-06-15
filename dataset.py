import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image
import os
from xml.dom.minidom import parse
from utils import *

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Mydataset(Dataset):
    def __init__(self, folder, label, transform=None):
        self.foldername = folder
        self.data = []
        self.label = label      # 1:MSI,  0:MSS
        self.transform = transform
        self.img_path_list = list()
        for data_name in os.listdir(folder):
            path = os.path.join(self.foldername, data_name)
        
            self.img_path_list.append(path)

        # for data_name in os.listdir(folder):
        #     path = os.path.join(self.foldername, data_name)
        #     img_path = [os.path.join(path, img_name) for img_name in os.listdir(path)]
        #     self.img_path_list.extend(img_path)

            
        # self.img_name_list = os.listdir(folder)
#         self.img_name_list = sorted(self.img_name_list)
        

    def get_img_name(self, index):
        return self.img_path_list[index]

    def __getitem__(self, index):
        path = self.img_path_list[index]
        #path = self.foldername + '/' + name
        # print(path)

        img = Image.open(path).convert('RGB')
        # print(img.size)

        if self.transform is not None:
            img = self.transform(img)

        label = self.label
        return img, label

    def __len__(self):
        return len(self.img_path_list)


    #def load(self):
        
        #print(self.data[0][1][0])

