#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:58:54 2017

@author: gregorymckay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:29:29 2017

@author: gregorymckay
"""

import os, os.path
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import pandas as pd
from PIL import ImageOps
from PIL import Image
#import imageio
import time
import random
from random import randint
import csv
#import cv2
from skimage import io, transform
from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS.append('tif')
#class ContrastiveLoss(nn.Module):
#
#    def __init__(self, margin=1.0):
#        super(ContrastiveLoss, self).__init__()
#        self.margin = margin
#
#    def forward(self, output, label):
#        loss_contrastive = torch.mean((label) * torch.pow(output, 2) +
#                                      (1-label) * torch.pow(torch.clamp(self.margin - output, min=0.0), 2))
#
#        return loss_contrastive

# 241x241x3 channel input

class cvcDataset(Dataset):
    
    def __init__(self,imageFolderDataset, transform=None, augmentation=False):
        self.imageFolderDataset = imageFolderDataset  
        self.transform = transform
        self.augmentation = augmentation
        
    def __getitem__(self,index):
        path1 = self.imageFolderDataset.root + '/Cropped/' + str(index + 1) + 'c.tif'
        path2 = self.imageFolderDataset.root + '/GT_Cropped/' + str(index + 1) + 'c.tif'
#        img1 = io.imread(path1)
#        img2 = io.imread(path2)
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        
#        img1 = transform.resize(img1, (240, 240), preserve_range=True, mode='reflect')
#        img2 = transform.resize(img2, (240, 240), preserve_range=True, mode='reflect')
        
        img1 = img1.resize((240,240), Image.ANTIALIAS)
        img2 = img2.resize((240,240), Image.ANTIALIAS)
        
#        img2[img2 < 128] = 0
#        img2[img2 >= 128] = 255
        
        if self.augmentation:
            self.augmentation = False

            ## data augmentation
        # p = random.uniform(0, 1)
        # if p > 0.5:
        #     scale_1 = random.uniform(0.7, 1.3)
        #     scale_2 = random.uniform(0.7, 1.3)
        #     rotation = randint(-30,30)
        #     translation_x = randint(-10,10)
        #     translation_y = randint(-10,10)
        #     mirror_h = randint(0,1)
        #     mirror_v = randint(0,1)
        #
        #     img1 = img1.rotate(rotation)
        #     img2 = img2.rotate(rotation)
        #
        #     img1 = img1.transform(img1.size, Image.AFFINE, (scale_1, 0, translation_x, 0, scale_2, translation_y))
        #     img2 = img2.transform(img2.size, Image.AFFINE, (scale_1, 0, translation_x, 0, scale_2, translation_y))
        #
        #     if mirror_h == 1:
        #         img1 = ImageOps.mirror(img1)
        #         img2 = ImageOps.mirror(img2)
        #     if mirror_v == 1:
        #         img1 = ImageOps.flip(img1)
        #         img2 = ImageOps.flip(img2)
                
        img1 = np.array(img1, dtype='uint8')
        im_gt_npar = np.array(img2, dtype='uint8')
        
        im_gt_npar[im_gt_npar < 128] = 0
        im_gt_npar[im_gt_npar >= 128] = 255      
        
#        nimg1 = np.array(img1, dtype='uint8')
#        nimg1 = cv2.cvtColor(nimg1, cv2.COLOR_RGB2GRAY)
#        nimg1 = np.array(nimg1, dtype='uint8')
#        SbS = np.concatenate((nimg1, im_gt_npar), axis=1)
#        plt.imshow(SbS)
#        plt.show()         
      
        im_gt_ones = im_gt_npar==255
        im_gt_ones = im_gt_ones.astype(int)
        im_gt_zeros = im_gt_npar==0
        im_gt_zeros = im_gt_zeros.astype(int)
        
        target_2d = np.zeros((240, 240, 2))
        target_2d[:, :, 0] = im_gt_ones
        target_2d[:, :, 1] = im_gt_zeros
        gt = target_2d

        # Data augmentation Image 1
        
        img1 = torch.from_numpy(img1)
        img1 = img1.permute(2,0,1)
        img1 = img1.float()
        
        gt = torch.from_numpy(gt)
        gt = gt.permute(2,0,1)
        gt = gt.float()

        if self.transform is not None:
            img1 = self.transform(img1)
            gt = self.transform(gt)
        
        return img1, gt
    
    def __len__(self):
        DIR = 'CVC_Data_Train/Cropped'
        return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and '.tif' in name])
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # start with 240
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2) # 120
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # 60
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # 30
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # 15
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)   
        
        self.conv11 = nn.ConvTranspose2d(1024, 512, 2, stride=2) # 30
        self.conv12 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv14 = nn.ConvTranspose2d(512, 256, 2, stride=2) # 60
        self.conv15 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv17 = nn.ConvTranspose2d(256, 128, 2, stride=2) # 120
        self.conv18 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv19 = nn.Conv2d(128, 128, 3, padding=1)      

        self.conv20 = nn.ConvTranspose2d(128, 64, 2, stride=2) # 240
        self.conv21 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1) 
        
        self.conv23 = nn.Conv2d(64, 2, 1)
        
        self.sigmoid = nn.Sigmoid()
        
#    def center_crop(self, layer, target_size):
#        batch_size, n_channels, layer_width, layer_height = layer.size()
#        xy1 = (layer_width - target_size) // 2
#        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]
    
    def forward(self, x):      
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x1_dup = x
        x = self.pool1(x)
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x2_dup = x
        x = self.pool2(x) 
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x3_dup = x
        x = self.pool3(x)  
        x = F.relu(self.conv8(F.relu(self.conv7(x))))
        x4_dup = x
        x = self.pool4(x)          
        x = F.relu(self.conv10(F.relu(self.conv9(x))))
        
        x = self.conv11(x)
        x = torch.cat((x4_dup,x),1)

        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.conv14(x)
        x = torch.cat((x3_dup,x),1)
      
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.conv17(x)
        x = torch.cat((x2_dup,x),1)       
      
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = self.conv20(x)
        x = torch.cat((x1_dup,x),1)    
       
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.conv23(x)
        
        x = self.sigmoid(x)
        return x


# trans = transforms.Compose([transforms.ToTensor()])
#dataset_folder = dset.ImageFolder(root='CVC_Data_Train')
dataset_folder = dset.ImageFolder(root='CVC_Data_Train')
dataset = cvcDataset(imageFolderDataset=dataset_folder)
batchsize = 16
epochs = 60
learning_rate = 0.00005

dataset_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=8)

#net2 = Net().cuda()
net = Net().cuda()
#net = Net()
#net.train(False) # Used in the case of badtch size 1, otherwise BatchNorm returns 

optimizer = optim.Adam(net.parameters(),lr=learning_rate)
lossfn = torch.nn.BCELoss()
loss_log = []
counter = []
loss_history = [] 
iteration_number= 0
i = 1
for epoch in range(epochs):
    for data in dataset_loader:
        image1, gt = data
        image1, gt = Variable(image1).cuda(), Variable(gt).cuda()
#        image1, gt = Variable(image1), Variable(gt)
#        im = gt.data.numpy()[0, 0, :, :]*255
#        print(im.shape)
#        plt.imshow(im)
#        plt.show
        output = net(image1)
        optimizer.zero_grad()
        loss = lossfn(output, gt)
        loss.backward()
        optimizer.step()
        loss_log.append(loss.data[0])
        i+=1
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss.data[0])    
            
#    print 'epoch = ', epoch, ', loss = ', loss.data[0]

torch.save(net.state_dict(), 'unet_batchsize16_60epochs_5re-5')
