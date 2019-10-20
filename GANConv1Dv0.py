#!/usr/bin/env python
# coding: utf-8
# In[ ]:


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape,self).__init__()
        self.shape = shape
    
    def forward(self,input):
        return input.view(input.size(0), *self.shape)
    
# Generator Network
# Input: Noise input as pytorch tensor. Shape: [batch_size,25].
# Output: Batch of artificial power consumption curves as pytorch tensor. Shape: [batch_size,1,336].

class GeneratorConv1D(nn.Module):
    def __init__(self):
        super(GeneratorConv1D, self).__init__()
        
        def blockConv(in_channels,out_channels,kernel_size,stride,padding,out_padding):
            Layers = [nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=out_padding)]
            Layers.append(nn.ReLU())
            return Layers
        
        def blockFC(in_features,out_features):
            Layers = [nn.Linear(in_features,out_features)]
            Layers.append(nn.ReLU())
            return Layers
            
        self.model = nn.Sequential(
        *blockFC(25,5632),
        Reshape([256,22]),
        *blockConv(256,128,5,2,2,0),
        *blockConv(128,64,5,2,2,0),
        *blockConv(64,32,5,2,2,0),
        *blockConv(32,16,5,2,2,0),
        nn.ConvTranspose1d(16, 1, 6, stride=1, padding=3,
                                     output_padding=0),
        nn.Tanh()
        )

    def forward(self,z):
        TimeSeries = self.model(z)
        return TimeSeries

# Discriminator Network
# Input: Power consumption curves as pytorch tensor. Shape: [batch_size,1,336].
# Output: Probability of input belonging to the training set as pytorch tensor. Shape: [batch_size].
            
class DiscriminatorConv1D(nn.Module):
    def __init__(self):
        super(DiscriminatorConv1D, self).__init__()
        
        def blockConv(in_channels,out_channels,kernel_size,stride,padding):
            Layers = [nn.utils.spectral_norm(nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding))]
            Layers.append(nn.LeakyReLU(0.2))
            return Layers
        
        def blockFC(in_features,out_features):
            Layers = [nn.utils.spectral_norm(nn.Linear(in_features,out_features))]
            Layers.append(nn.LeakyReLU(0.2))
            return Layers
        
        self.model = nn.Sequential(
        *blockConv(1,16,4,2,2),
        *blockConv(16,32,5,2,2),
        *blockConv(32,64,5,2,2),
        *blockConv(64,128,5,2,2),
        Flatten(),
        nn.utils.spectral_norm(nn.Linear(2816,1))
#         nn.Sigmoid()
        )
        
    def forward(self,z):
        validity = self.model(z)
        return validity
