#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Input: four frames of the Atari game
# Output: one saliency map for input episode

'''
One incident: (data, target) 
  Data: 4*num_pixel*num_pixel 
  Target: 1*num_pixel*num_pixel
Data set: 
  num_sample*(data, target)
'''


# In[2]:


# Imports
import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# In[117]:



# Input: four frames of the Atari game
# Output: one saliency map for input episode

'''
One incident: (data, target) 
  Data: 4*num_pixel*num_pixel 
  Target: 1*num_pixel*num_pixel
Data set: 
  num_sample*(data, target)
'''

# Imports

import torch
import torch.nn as nn

class Conv_Deconv(nn.Module):
    def __init__(self):
        super().__init__()
        # super(Conv_Deconv, self).__init__()
        #image_width = 84  # TODO 84*84 input size
        #image_height = 84
        #n_frames = 4, in dynamic CNN these are channels 
        #self.in_dim = image_width * image_length * in_frames
        self.conv1 = nn.Conv2d(in_channels=4,out_channels=64, kernel_size=3, padding=1)  # TODO modify inchannel
        
        self.residual1 = nn.Sequential(
          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 1),
          torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
          torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 1),
          torch.nn.BatchNorm2d(64),
        )
    
        self.residual2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 2),
            torch.nn.BatchNorm2d(64),
        )

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),  # TODO conv2d doens't take batch_size as input
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
            
        self.inception1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, return_indices=True)
            #torch.nn.Flatten(),
            #torch.nn.Linear(160*210*256, 256)
        )
        '''
        self.inception2 = torch.nn.Sequential(
            torch.nn.Conv2(in_channels = 256, out_channels = 256, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Conv2(in_channels = 256, out_channels = 256, kernel_size = 5, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            #torch.nn.Flatten(),
            #torch.nn.Linear(160*210*256, 256)
            
            )
        self.inception3 = torch.nn.Sequential(
            torch.nn.Conv2(in_channels = 256, out_channels = 256, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            #torch.nn.Flatten(),
            #torch.nn.Linear(160*210*256, 256)
            )
        ''' 
        #self.inception_before = torch.cat((self.inception1, self.inception2, self.inception3), dim=1)  # TODO need to be in forward
        #self.inception_after = torch.Linear(160*210*256*3, 160*210*256)
        self.inception1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, return_indices=True)
            #torch.nn.Flatten(),
            #torch.nn.Linear(160*210*256, 256)
        )
        self.inter = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            #torch.nn.BatchNorm2d(256),
            #torch.nn.ReLU(),
            #torch.nn.MaxUnpool2d(kernel_size=2, stride=2),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.decoder2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, return_indices=True),
            
            torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
            
        )
    

    def forward(self, x):  # x 4*84*84
        out1 = torch.relu(self.conv1(x))  # out1 size = 64*featuremaph*featuremapw DIM: 1*64*84*84
        out_res = torch.relu(self.residual1(out1) + self.residual2(out1))  # DIM res1: 1*64*88*88, DIM res2: 1*64*86*86
        # -> added padding to res2
        out2 = self.encoder(out1) # DIM: 1*256*10*10
        out2, indices = self.inception1(out2) # DIM: 1*256*6*6
        #print(indices.type) #DIM:1*256*12*12
        out2 = self.inter(out2, indices) # DIM: 1*256*12*12
        out2 = self.decoder1(x)
        #out2 = self.decoder1(out2, indices) 
        #out = torch.cat((out_res, out2), dim=0)
        #out = self.decoder2(out)
        #out = nn.torch.Softmax(dim=0)
        return out2 # predicted map 1*84*84

# TODO create test_data=tensor.ones(4*84*84)
#create the network
model = Conv_Deconv()
x = torch.ones(1,4,84,84) 
out = model.forward(x)
#print(out)
print(out.shape)


# How to make matching dimensions:
# - Increase or reduce padding
# - Add pooling layer

# In[ ]:




