#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:35:22 2017

@author: gregorymckay
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
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
import os

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
#        
        x = self.conv11(x)
        x = torch.cat((x4_dup,x),1)

        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.conv14(x)
        x = torch.cat((x3_dup,x),1)
##        
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.conv17(x)
        x = torch.cat((x2_dup,x),1)       
##        
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = self.conv20(x)
        x = torch.cat((x1_dup,x),1)    
       
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.conv23(x)
        
        x = self.sigmoid(x)
        return x


        
net = Net()
#net = Net().cuda()

# Load trained weights and biases here
#net.load_state_dict(torch.load('/Users/gregorymckay/Desktop/U-Net/unet_batchsize8_25epochs/unet_batchsize8_25epochs', map_location=lambda storage, loc: storage))
net.load_state_dict(torch.load('./unet_batchsize16_100epochs_5re-5', map_location=lambda storage, loc: storage))

net.train(False) # Used in the case of batch size 1 with batch norm
net.eval()


#criterion = nn.BCELoss()
#optimizer = optim.Adam(net.parameters(), lr=0.00001)
#
#
#counter = []
#loss_history = [] 
#iteration_number= 0


# Data Loading
#num_files = (len([name for name in os.listdir('Cropped/')])-1)
num_files = (len([name for name in os.listdir('./CVC_Data_Test/Cropped/')])-1)
n_train = num_files
minibatch_size = n_train
#rand_im_ind = np.random.randint(1,n_train, size = minibatch_size)

#for epoch in range(15):  # loop over the dataset multiple times
#    running_loss = 0.0
#for i in range(500,613):
#for i in range(1,num_files):
for i in range(500,613):
#        im_1 = Image.open('Cropped/' + str(i) + 'c.tif')
    im_1 = Image.open('./CVC_Data_Test/Cropped/' + str(i) + 'c.tif')
    #im_1 = Image.open('./99c.tif')
    im_gt = Image.open('./CVC_Data_Test/GT_Cropped/' + str(i) + 'c.tif')
    
    # Colonoscopy Image rescaling to 572x572
    width, height = im_1.size   
    new_width = 240
    new_height = 240
    im_1 = im_1.resize((new_width,new_height), Image.ANTIALIAS)
    
    # Colonoscopy image to tensor
    im_1_npar = np.array(im_1)
    im_1_tensor = torch.from_numpy(im_1_npar)
    im_1_tensor = im_1_tensor.unsqueeze(0)
    im_1_tensor = im_1_tensor.permute(0,3,1,2)
    inputs_im_1 = Variable(im_1_tensor.float(), requires_grad=True)
    #        inputs_im_1 = Variable(im_1_tensor.float(), requires_grad=True).cuda()
    
    # Ground truth binary image rescaling to 572x572 and then 388x388 center cropped
    im_gt = im_gt.resize((new_width,new_height), Image.ANTIALIAS)
#        width_gt, height_gt = im_gt.size
#        new_width_gt = 388
#        new_height_gt = 388
#        left = (width_gt - new_width_gt)/2
#        top = (height_gt - new_height_gt)/2
#        right = (width_gt + new_width_gt)/2
#        bottom = (height_gt + new_height_gt)/2
#        im_gt = im_gt.crop((left, top, right, bottom))
    
    # Ground truth binary image to 2x388x388 tensor
    im_gt_npar = np.array(im_gt)
    im_gt_ones = im_gt_npar==255
    im_gt_ones = im_gt_ones.astype(int)
    im_gt_zeros = im_gt_npar==0
    im_gt_zeros = im_gt_zeros.astype(int)
    
    target_2d = np.zeros((2,240,240))
    target_2d[0] = im_gt_ones
    target_2d[1] = im_gt_zeros
    target_tensor = torch.from_numpy(target_2d)
    
    target = Variable(target_tensor.float())
    #        target = Variable(target_tensor.float()).cuda()
    
    
    output = net(inputs_im_1)
    
    threshold = 0.5
    
    polyp = (output.data.numpy()[0][0] - output.data.numpy()[0][1] + np.ones(output.data.numpy()[0][1].shape)) / 2
    polyp[polyp >= threshold] = 255
    polyp[polyp < threshold] = 0
    polyp = polyp.astype(np.uint8)
    
#    background = output.data.numpy()[0][1]
#    background[background >= threshold] = 255
#    background[background < threshold] = 0
#    background = background.astype(np.uint8)
#    
#    polyp_final_arr = background + polyp
#    polyp_final_arr[polyp_final_arr > 1] = 255
#
#    polyp_final_arr = polyp_final_arr.astype(np.uint8)

    img_polyp = Image.fromarray(polyp,'L')   
#    img_polyp = Image.fromarray(polyp_final_arr,'L')       
    SbS = np.zeros((240,480))
    x_SbS = len(SbS[:,1])
    y_SbS = len(SbS[1,:])  
    x = len(im_gt_npar[:,1])
    y = len(im_gt_npar[1,:])    
    
    SbS[0:x,0:y] = im_gt_npar
#    SbS[0:x,y:2*y] = polyp_final_arr
    SbS[0:x,y:2*y] = polyp
    SbS = SbS.astype(np.uint8)
    im_SbS = Image.fromarray(SbS,'L')
    im_SbS.save('./CVC Data/UNet_V4_Batchsize16_100epochs_5e-5lr/'+str(i)+'c.tif')
    print(i)
#    im_gt.save('/Users/gregorymckay/Desktop/unet_14epochs_test/GT_Cropped388x388/'+str(i)+'c.tif')
#    img_polyp.save('/Users/gregorymckay/Desktop/unet_14epochs_test/NN_Output388x388'+str(i)+'c.tif')
    
#    optimizer.zero_grad()
#    loss = criterion(output[0], target)
#    loss.backward()
#    optimizer.step()
        
#        if i %10 == 0 :
#            print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
#            iteration_number +=10
#            counter.append(iteration_number)
#            loss_history.append(loss.data[0])