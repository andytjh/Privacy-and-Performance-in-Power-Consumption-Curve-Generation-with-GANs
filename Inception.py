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

# Inception Cell for the generator
    
class InceptionG(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionG,self).__init__()
        
        self.large = nn.Sequential(
#         nn.ReflectionPad1d(24),
        nn.Conv1d(in_channels,out_channels,49,stride=1,padding=24)
        )
        
        self.sparse1 = nn.Sequential(
#         nn.ReflectionPad1d(144),
        nn.Conv1d(in_channels,out_channels,7,stride=1,padding=144,dilation=48)
        )
            
        self.sparse2 = nn.Sequential(
#         nn.ReflectionPad1d(40),
        nn.Conv1d(in_channels,out_channels,6,stride=1,padding=40,dilation=16)
        )
        
        self.sparse3 = nn.Sequential(
#         nn.ReflectionPad1d(20),
        nn.Conv1d(in_channels,out_channels,6,stride=1,padding=20,dilation=8)
        )
        
        self.Batch_norm1 = nn.Sequential(
        nn.BatchNorm1d(out_channels)
        )
        
    def forward(self,x):
        branch1 = self.large(x)
        branch2 = self.sparse1(x)
        branch3 = self.sparse2(x)
        branch4 = self.sparse3(x)
        return self.Batch_norm1(branch1 + branch2 + branch3 + branch4)

# Residual Cell for the generator
    
class ResidualG(InceptionG):
    def __init__(self,in_channels,out_channels):
        super(ResidualG,self).__init__(in_channels,out_channels)
    
    def forward(self,x):
        branch1 = self.large(x)
        branch2 = self.sparse1(x)
        branch3 = self.sparse2(x)
        branch4 = self.sparse3(x)
        inception = self.Batch_norm1(branch1 + branch2 + branch3 + branch4)
        return torch.cat((x,inception),1)

# Generator Network
# Input: Noise input as pytorch tensor. Shape: [batch_size,42].
# Output: Batch of artificial power consumption curves as pytorch tensor. Shape: [batch_size,1,336].
    
class GeneratorConv1D(nn.Module):
    def __init__(self):
        super(GeneratorConv1D, self).__init__()
        
        def blockFC(in_features,out_features):
            Layers = [nn.Linear(in_features,out_features)]
            Layers.append(nn.BatchNorm1d(out_features))
            Layers.append(nn.ReLU())
            return Layers
        
        def blockUpConv1d(in_channels,out_channels,kernel_size):
            Layers = [nn.ConvTranspose1d(in_channels,out_channels,kernel_size,dilation=1,stride=1,padding=0)]
            Layers.append(nn.BatchNorm1d(out_channels))
            Layers.append(nn.ReLU())
            return Layers
        
        self.model = nn.Sequential(
        *blockFC(42,2688),
        Reshape([8,336]),
        ResidualG(8,16),
        nn.ReLU(),
        InceptionG(24,32),
        nn.ReLU(),
        InceptionG(32,32),
        nn.ReLU(),
        ResidualG(32,16),
        nn.ReLU(),
        InceptionG(48,64),
        nn.ReLU(),
        InceptionG(64,64),
        nn.ReLU(),
        ResidualG(64,16),
        nn.ReLU(),
        InceptionG(80,96),
        nn.ReLU(),
        InceptionG(96,1),
        nn.Tanh()
        )

    def forward(self,z):
        TimeSeries = self.model(z)
        return TimeSeries
    
# Inception Cell for the discriminator

class InceptionD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionD,self).__init__()
        
        self.large = nn.Sequential(
#         nn.ReflectionPad1d(24),
        nn.utils.spectral_norm(nn.Conv1d(in_channels,out_channels,49,stride=1,padding=24))
        )
        
        self.sparse1 = nn.Sequential(
#         nn.ReflectionPad1d(144),
        nn.utils.spectral_norm(nn.Conv1d(in_channels,out_channels,7,stride=1,padding=144,dilation=48))
        )
            
        self.sparse2 = nn.Sequential(
#         nn.ReflectionPad1d(40),
        nn.utils.spectral_norm(nn.Conv1d(in_channels,out_channels,6,stride=1,padding=40,dilation=16))
        )
        
        self.sparse3 = nn.Sequential(
#         nn.ReflectionPad1d(20),
        nn.utils.spectral_norm(nn.Conv1d(in_channels,out_channels,6,stride=1,padding=20,dilation=8))
        )
        
    def forward(self,x):
        branch1 = self.large(x)
        branch2 = self.sparse1(x)
        branch3 = self.sparse2(x)
        branch4 = self.sparse3(x)
        return branch1 + branch2 + branch3 + branch4

# Residual Cell for the discriminator
    
class ResidualD(InceptionD):
    def __init__(self,in_channels,out_channels):
        super(ResidualD,self).__init__(in_channels,out_channels)
    
    def forward(self,x):
        branch1 = self.large(x)
        branch2 = self.sparse1(x)
        branch3 = self.sparse2(x)
        branch4 = self.sparse3(x)
        inception = branch1 + branch2 + branch3 + branch4
        return torch.cat((x,inception),1)
    
# Discriminator Network
# Input: Power consumption curves as pytorch tensor. Shape: [batch_size,1,336].
# Output: Probability of input belonging to the training set as pytorch tensor. Shape: [batch_size].
            
class DiscriminatorConv1D(nn.Module):
    def __init__(self):
        super(DiscriminatorConv1D, self).__init__()
        
        def blockFC(in_features,out_features):
            Layers = [nn.utils.spectral_norm(nn.Linear(in_features,out_features))]
            Layers.append(nn.ReLU())
            return Layers
        
        self.model = nn.Sequential(
        ResidualD(1,16),
        nn.ReLU(),
        ResidualD(17,16),
        nn.ReLU(),
        ResidualD(33,16),
        nn.ReLU(),
        ResidualD(49,32),
        nn.ReLU(),
        ResidualD(81,32),
        nn.ReLU(),
        ResidualD(113,32),
        nn.ReLU(),
        InceptionD(145,1),
        nn.ReLU(),
        Flatten(),
        *blockFC(336,168),
        *blockFC(168,84),
        nn.utils.spectral_norm(nn.Linear(84,1))
#         nn.Sigmoid()
        )
        
    def forward(self,z):
        validity = self.model(z)
        return validity
    