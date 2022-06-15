import torch
import torchvision
import torch.nn as nn

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
train_on_gpu = torch.cuda.is_available() 

MSI_train_folder = "../../xiaohui/tumor_patch/train/MSI"
MSS_train_folder = "../../xiaohui/tumor_patch/train/MSS"

MSI_val_folder = "../../xiaohui/tumor_patch/val/MSI"
MSS_val_folder = "../../xiaohui/tumor_patch/val/MSS"

MSI_test_folder = "../../xiaohui/tumor_patch/test/MSI"
MSS_test_folder = "../../xiaohui/tumor_patch/test/MSS"


cat_train_folder = "cifar10data/train/cat"
dog_train_folder = "cifar10data/train/dog"

cat_val_folder = "cifar10data/test/cat"
dog_val_folder = "cifar10data/test/dog"

epsilon = 0.031
alpha = 0.007
attack_iters = 10
lower_limit = 0
upper_limit = 1


data_folder = 'data/'
min_num = 10
# n_epochs = 30
