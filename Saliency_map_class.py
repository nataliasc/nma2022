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

import torch
import torch.nn as nn


# In[14]:


class Conv_Deconv(nn.Module):
    def __init__(self):
        super().__init__()
        # super(Conv_Deonv, self).__init__()
        #image_width = 160  # TODO 84*84 input size
        #image_height = 210
        #n_frames = 4
        #self.in_dim = image_width * image_length * batch_size
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(3, 3), padding=1)  # TODO modify inchannel
        self.residual1 = nn.Sequential(
          torch.nn.Conv2d(batch_size = 4, in_channels = 64, out_channels = 64, kernel_size = 1, padding = 1),
          torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
          torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 1),
          torch.nn.BatchNorm2d(64),
          #torch.nn.Flatten(),
          #torch.nn.Linear(160*210*64, 64)
        )
    
        self.residual2 = nn.Sequential(
            torch.nn.Conv2d(batch_size = 4, in_channels = 64, out_channels = 64, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(64),
            #torch.nn.Flatten(),
            #torch.nn.Linear(160*210*64, 64)
        )
    
        self.residual = torch.ReLU(self.residual1 + self.residual2)  # TODO this is in the wrong place

        #self.bn2 = nn.BatchNorm2d(256)
        self.encoder = torch.nn.Sequential(
            #Input = 1 x 160 x 210, Output = 96 x 160 x 210
            torch.nn.Conv2d(batch_size = 4, in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1),  # TODO conv2d doens't take batch_size as input
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2),

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
            torch.nn.Conv2(in_channels = 256, out_channels = 256, kernel_size = 1, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            
            torch.nn.Conv2(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            #torch.nn.Flatten(),
            #torch.nn.Linear(160*210*256, 256)
        )
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
            
        self.inception_before = torch.cat((self.inception1, self.inception2, self.inception3), dim=1)  # TODO need to be in forward
        self.inception_after = torch.Linear(160*210*256*3, 160*210*256)
        
			#torch.nn.Flatten(),
			#torch.nn.Linear(64*4*4, 512),
			#torch.nn.ReLU(),
			#torch.nn.Linear(512, 10)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.MaxUnpool2d(kernel_size=256, stride=2, padding=1),
            
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.MaxUnpool2d(kernel_size=128, stride=2, padding=1),
            
            residual(...),
            
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.MaxUnpool2d(kernel_size=64, stride=2, padding=1)
            
            torch.nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  
)

    def forward(self, x):  # x 4*84*84
        out1 = torch.relu(self.conv1(x))  # out1 size = 64*featuremaph*featuremapw


        return self.model(x)   # predicted map 1*84*84


# Questions:
# - In the residual and inception module only one value is specified. Should I assume that the in_channels = out_channels = this value?
# - In the first branch of the residual module it has 32. I assume this means out_channels. If we have already passed through the first convolutional layer and already have out_channels = 64, what should we do?
# - The description of encoder and decoder doesn't match the table. Which one should I go with?
# - What sort of dimensions should the model have by the end of inception module?

# In[ ]:


# TODO create test_data=tensor.ones(4*84*84)
create the network


