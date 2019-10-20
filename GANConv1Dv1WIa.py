#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from QualityMetrics import Indicators

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
# Input: Noise input as pytorch tensor. Shape: [batch_size,42].
# Output: Batch of artificial power consumption curves as pytorch tensor. Shape: [batch_size,1,336].

class GeneratorConv1D(nn.Module):
    def __init__(self):
        super(GeneratorConv1D, self).__init__()
        
        def blockConv(in_channels,out_channels,kernel_size,stride,padding,out_padding):
            Layers = [nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=out_padding)]
            Layers.append(nn.BatchNorm1d(out_channels))
            Layers.append(nn.ReLU())
            return Layers
        
        def blockFC(in_features,out_features):
            Layers = [nn.Linear(in_features,out_features)]
            Layers.append(nn.BatchNorm1d(out_features))
            Layers.append(nn.ReLU())
            return Layers
            
        self.model = nn.Sequential(
        *blockFC(42,336),
        *blockFC(336,672),
        *blockFC(672,1344),
        Reshape([32,42]),
        *blockConv(32,32,8,2,24,0),
        *blockConv(32,16,8,2,3,0),
        *blockConv(16,8,8,2,3,0),
        nn.ConvTranspose1d(8, 1, 8, stride=2, padding=3,
                                     output_padding=0),
        nn.Tanh()
        )

    def forward(self,z):
        TimeSeries = self.model(z)
        return TimeSeries
    
# Discriminator Network: Signal
# Input: Power consumption curves as pytorch tensor. Shape: [batch_size,1,336].
# Output: Probability of input belonging to the training set as pytorch tensor. Shape: [batch_size].
            
class DiscriminatorSignal(nn.Module):
    def __init__(self):
        super(DiscriminatorSignal, self).__init__()
        
        def blockConv(in_channels,out_channels,kernel_size,stride,padding):
            Layers = [nn.utils.spectral_norm(nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding))]
            Layers.append(nn.ReLU())
            return Layers
        
        def blockFC(in_features,out_features):
            Layers = [nn.utils.spectral_norm(nn.Linear(in_features,out_features))]
            Layers.append(nn.ReLU())
            return Layers
        
        self.model = nn.Sequential(
        *blockConv(1,8,8,2,3),
        *blockConv(8,16,8,2,3),
        *blockConv(16,32,8,2,3),
        Flatten(),
        *blockFC(1344,672),
        *blockFC(672,336),
        nn.Linear(336,1)
#         nn.Sigmoid()
        )
        
    def forward(self,z):
        validity = self.model(z)
        return validity
    
# Discriminator Network: Indicators
# Input: Power consumption curves as pytorch tensor. Shape: [batch_size,1,336]. (The indicators are computed by the discriminator).
# Output: Probability of input belonging to the training set as pytorch tensor. Shape: [batch_size].

class DiscriminatorIndicators(nn.Module):
    def __init__(self):
        super(DiscriminatorIndicators, self).__init__()
        
        def blockFC(in_features,out_features):
            Layers = [nn.utils.spectral_norm(nn.Linear(in_features,out_features))]
            Layers.append(nn.LeakyReLU(0.2))
            return Layers
        
        self.model = nn.Sequential(
        nn.BatchNorm1d(5),
        *blockFC(5,5),
        *blockFC(5,5),
        *blockFC(5,5),
        nn.utils.spectral_norm(nn.Linear(5,1))
#         nn.Sigmoid()
        )
            
    def forward(self,signal):
        indicators = Indicators(signal)
        validity = self.model(indicators)
        return validity
