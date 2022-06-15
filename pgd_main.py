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
def get_dataloader(batch_size):
#     dataset = raw_diff_Data
    train_dataset = train_MSI_Data + train_MSS_Data
    val_dataset = val_MSI_Data + val_MSS_Data

    
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2) 
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True, num_workers=2) 
    return train_dataloader, val_dataloader


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
def test_model(model, data_loader):
    criterion = nn.CrossEntropyLoss()                     #交叉熵损失  
    num=0
    accu=0
    Loss=0
    
    accu1 = 0
    accu2 = 0
    
    num1 = 0
    num2 = 0
    print("test: ")
    with torch.no_grad():
        for data,target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)   
    #         print(data.shape, data)
            output = model(data)  
    #         print(output.shape, output)
            loss=criterion(output,target)
            Loss += loss.item()

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            accu += (pred_y == y_label).sum()
    #         print(pred_y, y_label)
            # for i in range(len(y_label)):
            #     if y_label[i] == 1 and pred_y[i] == 1:
            #         accu1 += 1
            #     if y_label[i] == 0 and pred_y[i] == 0:
            #         accu2 += 1
            # num1 += (y_label == 1).sum()
            # num2 += (y_label == 0).sum()
    #         print(accu1 / num1, accu2 / num2)

            num += len(y_label)
    accu /= num
    # sensitivity = accu1 / num1
    # specificity = accu2 / num2
    Loss /= num
    print("loss:",Loss,"accuracy:",accu)
    # print("loss:",Loss,"accuracy:",accu, "sensitivity:", sensitivity, "specificity", specificity)
    return accu, Loss



def PGD_train_model(opts, cnn, train_dataloader, val_dataloader, model_name):
    criterion = nn.CrossEntropyLoss()                     #交叉熵损失                 
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20],gamma = 0.5)

    Train_accuracy = list()      #训练集正确率列表
    Val_accuracy = list()        #验证集正确率列表
    Train_loss = list()
    Val_loss=list()
    
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
                                                  ###启用 BatchNormalization 和 Dropout
        num=0
        accu=0
        Loss=0
        
        grad_scalar = GradScaler()
        for data, target in tqdm(train_dataloader):       
            data, target = data.to(device), target.to(device)    
            with autocast():
                cnn.eval()  
                delta = attack_pgd_full_linf(cnn, data, target, nn.CrossEntropyLoss(), epsilon, alpha, 
                                            attack_iters, lower_limit, upper_limit)
                cnn.train() 
                adv_data = data + delta
                adv_data = torch.clamp(adv_data, lower_limit, upper_limit)
                output = cnn(adv_data)                                                     
                loss = criterion(output, target)      
            Loss += loss.item()  
            optimizer.zero_grad()                           ##loss关于weight的导数置零
            grad_scalar.scale(loss).backward()                               ###反向传播
            grad_scalar.step(optimizer)                      #更新参数
            grad_scalar.update()
                

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            accu += (pred_y == y_label).sum()
    #         print(pred_y, y_label)


            num += len(y_label)

            #train_loss += loss.item()*data.size(0)
        scheduler.step()
        cnn.eval()  
        train_acc = accu / num
        train_loss = Loss / num
#         if epoch % 10 == 0:
        torch.save(cnn.state_dict(),f'checkpoints_model/{model_name}/random_init/{opts.model}_parameter.pkl') 
    
        val_acc, val_loss = test_model(cnn, val_dataloader)
        
        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            torch.save(cnn.state_dict(),f'checkpoints_model/{model_name}/random_init/{opts.model}_best_parameter.pkl') 
            
        
        Train_accuracy.append(train_acc)
        Val_accuracy.append(val_acc)
        Train_loss.append(train_loss)
        Val_loss.append(val_loss)
        print("epoch =",epoch,":\ntrain-- accuracy =",train_acc,"loss =",train_loss)
        print("epoch =",epoch,":\nval-- accuracy =",val_acc,"loss =",val_loss)
        
        
        fp = open(f'draw_result/{model_name}/random_init/{opts.model}_acc.csv','a+')
        acc_result = []
        acc_result.append(epoch)
        acc_result.append(train_acc)
        acc_result.append(val_acc)
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(acc_result)
        fp.close()
        
        fp = open(f'draw_result/{model_name}/random_init/{opts.model}_loss.csv','a+')
        loss_result = []
        loss_result.append(epoch)
        loss_result.append(train_loss)
        loss_result.append(val_loss)
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(loss_result)
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
    return model_name, model

        
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



train_MSI_Data = Mydataset(MSI_train_folder, 1, transform)
train_MSS_Data = Mydataset(MSS_train_folder, 0, transform)

val_MSI_Data = Mydataset(MSI_val_folder, 1, transform2)
val_MSS_Data = Mydataset(MSS_val_folder, 0, transform2)

# test_MSI_Data = Mydataset(MSI_test_folder, 1, transform2)
# test_MSS_Data = Mydataset(MSS_test_folder, 0, transform2)
parser = get_argparser()

opts = parser.parse_args()
model_name, cnn = get_model(opts)

n_epochs = opts.epochs
batch_size = opts.batch_size
learning_rate = opts.lr


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

    
train_dataloader, val_dataloader = get_dataloader(batch_size)
print(len(train_dataloader), len(val_dataloader))
# val_dataloader = get_val_data()


# opts = modify_command_options(opts)
if __name__ == '__main__':
    
    # train_k_fold_model(opts, cnn, model_name, 5)
    # train_model(opts, cnn, train_dataloader, val_dataloader, 'clean_model')
    PGD_train_model(opts, cnn, train_dataloader, val_dataloader, 'pgd_model')
#     print("train:")
#     test_model(cnn, train_dataloader)
#     print("val:")
#     test_model(cnn, val_dataloader)
    





