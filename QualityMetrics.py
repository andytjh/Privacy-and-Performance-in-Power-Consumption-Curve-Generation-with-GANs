#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch

if torch.cuda.is_available():
    cuda = True 
else: 
    cuda = False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# This function calculates the quality metrics (a.k.a. Indicators), namely, the mean, coefficient of variation, skewness, kurtosis and maximum-mean ratio, on a batch of curves.

def Indicators(Sample):
    # Input: 
    # -Sample: Batch of 1D samples as pytorch tensor. Shape: [batch_size,sample_length].
    # Output:
    # -Indicator_list: Pytorch tensor containing the indicators for each sample in the batch. Shape: [batch_size,5].
    
    epsilon = 1e-6 # Parameter to avoid division by 0.  
    if len(Sample.size()) > 2:
        Sample = torch.squeeze(Sample,1)
    mean1 = torch.mean(Sample,1)
    std1 = torch.std(Sample,1)
    CV = torch.div(std1,mean1+epsilon)
    maximum,_ = torch.max(Sample,1)
    maxmean = torch.div(maximum,mean1+epsilon)
    mean_aux = torch.unsqueeze(mean1,1)
    skewness = torch.div(torch.mean(torch.pow(Sample - mean_aux,3),1),torch.pow(std1,3)+epsilon)
    kurtosis = torch.div(torch.mean(torch.pow(Sample - mean_aux,4),1),torch.pow(std1,4)+epsilon)
    List = [mean1,skewness,CV,kurtosis,maxmean]
    return torch.stack(List,dim=1)
