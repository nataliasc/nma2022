#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


# In[28]:


'''
    One incident: (data, target) 
    Data: 4*num_pixel*num_pixel 
    Target: 1*num_pixel*num_pixel
    Data set: num_sample*(data, target)
'''


# In[29]:


class SimpleFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4),
            #torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            #torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=8, stride=4),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1) # if not working try dim=0
        
    def forward(self, x):  # x 4*84*84
        out = self.encoder(x) # 1*64*10*10
        out = self.decoder(out) # 1*1*25*25
        out = self.softmax(out) # 1*1*84*84
        return out # predicted map 1*84*84


# In[30]:


model = SimpleFCN()
x = torch.ones(1,4,84,84) 
out = model.forward(x)
print(out.shape)


# In[ ]:




