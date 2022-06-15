# from draw import draw_box
# import imp
from dataset import *
from utils import *
from Network import *
from argparser import *

import cv2
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import heapq
import random
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
# from sklearn.model_selection import KFold

from torch.cuda.amp import autocast, GradScaler

import csv
    

# print(torch.__version__)


# 获得训练集和验证集
def get_test_dataloader(batch_size):
#     dataset = raw_diff_Data
    test_dataset = test_MSI_Data + test_MSS_Data
    
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True, num_workers=2) 
    
    return test_dataloader


@torch.enable_grad()
def attack_pgd_full_linf(model, X, y, loss_fn,
                         epsilon, alpha, attack_iters, lower_limit, upper_limit):
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    for _ in range(attack_iters):
        # amp context beg
        # with autocast():
        output = model(X + delta)
        loss = loss_fn(output, y)
        loss.backward()
        # amp context end
        
        grad = delta.grad.detach()
        
        d = delta.data.detach()
        d = torch.clamp(d + alpha * torch.sign(grad), -epsilon, epsilon)
        d = torch.clamp(d, lower_limit - X, upper_limit - X)
        delta.data = d
        delta.grad.zero_()
    return delta.detach()


# 测试网络正确率
def test_model_clean(opts, cnn, test_dataloader, model_name, data_type):
    num=0
    accu=0

    accu1 = 0
    accu2 = 0
    
    num1 = 0
    num2 = 0
    cnn.eval()  
    print("test: ")
    with torch.no_grad():
        for data,target in tqdm(test_dataloader):
            data, target = data.to(device), target.to(device)   
    #         print(data.shape, data)
            output = cnn(data)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            accu += (pred_y == y_label).sum()
            for i in range(len(y_label)):
                if y_label[i] == 1 and pred_y[i] == 1:
                    accu1 += 1
                if y_label[i] == 0 and pred_y[i] == 0:
                    accu2 += 1
            num1 += (y_label == 1).sum()
            num2 += (y_label == 0).sum()
            # print(accu1 / num1, accu2 / num2)

            num += len(y_label)
    accu /= num
    sensitivity = accu1 / num1
    specificity = accu2 / num2

    print(data_type, "  accuracy:",accu, "sensitivity:", sensitivity, "specificity", specificity)

    
    fp = open(f'test_result/{model_name}/{opts.model}_result.csv','a+')
    acc_result = []
    acc_result.append(opts.checkpoint)
    acc_result.append(data_type)
    acc_result.append(accu)
    acc_result.append(sensitivity)
    acc_result.append(specificity)
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(acc_result)
    fp.close()
    return



def test_model_robust(opts, cnn, test_dataloader, model_name, data_type):
    num=0
    accu=0

    accu1 = 0
    accu2 = 0
    
    num1 = 0
    num2 = 0
    cnn.eval()  
    print("test: ")
    
    for data,target in tqdm(test_dataloader):
        data, target = data.to(device), target.to(device)    
        with autocast():
            delta = attack_pgd_full_linf(cnn, data, target, nn.CrossEntropyLoss(), epsilon, alpha, 
                                        attack_iters, lower_limit, upper_limit)
            adv_data = data + delta
            adv_data = torch.clamp(adv_data, lower_limit, upper_limit)
            output = cnn(adv_data)                                              
#         print(output.shape, output)

        pred_y = torch.max(output, 1)[1].cpu().numpy()
        y_label = target.cpu().numpy()
        accu += (pred_y == y_label).sum()

        for i in range(len(y_label)):
            if y_label[i] == 1 and pred_y[i] == 1:
                accu1 += 1
            if y_label[i] == 0 and pred_y[i] == 0:
                accu2 += 1
        num1 += (y_label == 1).sum()
        num2 += (y_label == 0).sum()
        # print(accu1 / num1, accu2 / num2)

        num += len(y_label)
    accu /= num
    sensitivity = accu1 / num1
    specificity = accu2 / num2

    print(data_type, "  accuracy:",accu, "sensitivity:", sensitivity, "specificity", specificity)

    
    fp = open(f'test_result/{model_name}/{opts.model}_result.csv','a+')
    acc_result = []
    acc_result.append(opts.checkpoint)
    acc_result.append(data_type)
    acc_result.append(accu)
    acc_result.append(sensitivity)
    acc_result.append(specificity)
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(acc_result)
    fp.close()
    
    return




def get_model(opts):
    if opts.model == 'resnet34':
        model = Resnet_Network()
        model_name = 'resnet'
        
    elif opts.model == 'vgg11':
        model = vgg_Network()
        model_name = 'vgg'

    else:
        model = Resnet_Network()
    return model

        
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor()
                            
])

transform2 = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor()
                            
])


transform3 = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.RandomCrop(224),  #先四周填充0，在吧图像随机裁剪成224*224
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor()
                            
])



test_MSI_Data = Mydataset(MSI_test_folder, 1, transform2)
test_MSS_Data = Mydataset(MSS_test_folder, 0, transform2)

parser = get_argparser()

opts = parser.parse_args()
cnn = get_model(opts)

n_epochs = opts.epochs
batch_size = opts.batch_size
learning_rate = opts.lr
model_name = opts.model_name

print(opts)

print(f'{opts.model}_parameter.pkl')

device = torch.device("cuda") 
if train_on_gpu:                                                   #部署到GPU上
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")



# print(cnn)

cnn = cnn.to(device)    

cnn = nn.DataParallel(cnn)
# torch.save(cnn, "net_params.pkl")
if opts.checkpoint != None:
    print(opts.checkpoint)
    cnn.load_state_dict(torch.load(opts.checkpoint))

    
test_dataloader = get_test_dataloader(batch_size)
print(len(test_dataloader))
# val_dataloader = get_val_data()


# opts = modify_command_options(opts)
if __name__ == '__main__':
    # fp = open(f'test_result/{model_name}/{opts.model}_result.csv','a+')
    # acc_result = []
    # acc_result.append('model')
    # acc_result.append('datatype')
    # acc_result.append('accuracy')
    # acc_result.append('sensitivity(MSI)')
    # acc_result.append('specificity(MSS)')
    # context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    # context.writerow(acc_result)
    # fp.close()

    test_model_clean(opts, cnn, test_dataloader, model_name, 'clean')
    test_model_robust(opts, cnn, test_dataloader, model_name, 'robust')
#     print("train:")
#     test_model(cnn, train_dataloader)
#     print("val:")
#     test_model(cnn, val_dataloader)
    





