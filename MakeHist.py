#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.stats import wasserstein_distance as EMD

# This function computes the empirical distributions of the indicators for natural (real) and artificial (fake) samples.

def MakeHist(indicators_real, indicators_fake):
    # Input: 
    # -Indicators_real: Numpy array of dimensions: [batch_size,5].
    # -Indicators_fake: Numpy array of dimensions: [batch_size,5].
    # Output:
    # -List_Hist_r: List of numpy arrays containing the probability distribution (height of the bin, normalized) for the indicators of the real samples. Shape: [5][bins_W]
    # -List_Hist_f: List of numpy arrays containing the probability distribution (height of the bin, normalized) for the indicators of the fake samples. Shape: [5][bins_W]
    # -List_Hist_x: List of numpy arrays containing the values (position of the bins) for the indicators of the real and fake samples. Shape: [5][bins_W]
    # -List_EMD: List of 5 real values corresponding to the Earth's Movers Distances between real and fake indicators.
    # -Avg_Ind_Index: Average Indicator Distance between fake and real samples (mean of List_EMD).
    
    num_samples = indicators_real.shape[0]
    
    mean_real = indicators_real[:,0]
    skewness_real = indicators_real[:,1]
    CV_real = indicators_real[:,2]
    kurtosis_real = indicators_real[:,3]
    maxmean_real = indicators_real[:,4]

    mean_fake = indicators_fake[:,0]
    skewness_fake = indicators_fake[:,1]
    CV_fake = indicators_fake[:,2]
    kurtosis_fake = indicators_fake[:,3]
    maxmean_fake = indicators_fake[:,4]
    
    bins_W = 51 #Number of bins in the histogram plus one

    mean_min = np.minimum(mean_real.min(),mean_fake.min())
    mean_max = np.maximum(mean_real.max(),mean_fake.max())
    Hist_mean_real = np.array(np.histogram(mean_real,bins=np.linspace(mean_min,mean_max,num=bins_W)))
    Hist_mean_fake = np.array(np.histogram(mean_fake,bins=np.linspace(mean_min,mean_max,num=bins_W)))
    
    N_mean = (bins_W-1)/((mean_max-mean_min)*num_samples)
    Hist_mean_real_y = Hist_mean_real[0]*N_mean
    Hist_mean_real_x = Hist_mean_real[1]
    Hist_mean_real_x = (Hist_mean_real_x[0:bins_W-1]+Hist_mean_real_x[1:bins_W])/2
    Hist_mean_fake_y = Hist_mean_fake[0]*N_mean
    Hist_mean_fake_x = Hist_mean_real_x

    skewness_min = np.minimum(skewness_real.min(),skewness_fake.min())
    skewness_max = np.maximum(skewness_real.max(),skewness_fake.max())
    Hist_skewness_real = np.array(np.histogram(skewness_real,bins=np.linspace(skewness_min,skewness_max,num=bins_W)))
    Hist_skewness_fake = np.array(np.histogram(skewness_fake,bins=np.linspace(skewness_min,skewness_max,num=bins_W)))

    N_skewness = (bins_W-1)/((skewness_max-skewness_min)*num_samples)
    Hist_skewness_real_y = Hist_skewness_real[0]*N_skewness
    Hist_skewness_real_x = Hist_skewness_real[1]
    Hist_skewness_real_x = (Hist_skewness_real_x[0:bins_W-1]+Hist_skewness_real_x[1:bins_W])/2
    Hist_skewness_fake_y = Hist_skewness_fake[0]*N_skewness
    Hist_skewness_fake_x = Hist_skewness_real_x

    CV_min = np.minimum(CV_real.min(),CV_fake.min())
    CV_max = np.maximum(CV_real.max(),CV_fake.max())
    Hist_CV_real = np.array(np.histogram(CV_real,bins=np.linspace(CV_min,CV_max,num=bins_W)))
    Hist_CV_fake = np.array(np.histogram(CV_fake,bins=np.linspace(CV_min,CV_max,num=bins_W)))

    N_CV = (bins_W-1)/((CV_max-CV_min)*num_samples)
    Hist_CV_real_y = Hist_CV_real[0]*N_CV
    Hist_CV_real_x = Hist_CV_real[1]
    Hist_CV_real_x = (Hist_CV_real_x[0:bins_W-1]+Hist_CV_real_x[1:bins_W])/2
    Hist_CV_fake_y = Hist_CV_fake[0]*N_CV
    Hist_CV_fake_x = Hist_CV_real_x

    kurtosis_min = np.minimum(kurtosis_real.min(),kurtosis_fake.min())
    kurtosis_max = np.maximum(kurtosis_real.max(),kurtosis_fake.max())
    Hist_kurtosis_real = np.array(np.histogram(kurtosis_real,bins=np.linspace(kurtosis_min,kurtosis_max,num=bins_W)))
    Hist_kurtosis_fake = np.array(np.histogram(kurtosis_fake,bins=np.linspace(kurtosis_min,kurtosis_max,num=bins_W)))

    N_kurtosis = (bins_W-1)/((kurtosis_max-kurtosis_min)*num_samples)
    Hist_kurtosis_real_y = Hist_kurtosis_real[0]*N_kurtosis
    Hist_kurtosis_real_x = Hist_kurtosis_real[1]
    Hist_kurtosis_real_x = (Hist_kurtosis_real_x[0:bins_W-1]+Hist_kurtosis_real_x[1:bins_W])/2
    Hist_kurtosis_fake_y = Hist_kurtosis_fake[0]*N_kurtosis
    Hist_kurtosis_fake_x = Hist_kurtosis_real_x

    maxmean_min = np.minimum(maxmean_real.min(),maxmean_fake.min())
    maxmean_max = np.maximum(maxmean_real.max(),maxmean_fake.max())
    Hist_maxmean_real = np.array(np.histogram(maxmean_real,bins=np.linspace(maxmean_min,maxmean_max,num=bins_W)))
    Hist_maxmean_fake = np.array(np.histogram(maxmean_fake,bins=np.linspace(maxmean_min,maxmean_max,num=bins_W)))

    N_maxmean = (bins_W-1)/((maxmean_max-maxmean_min)*num_samples)
    Hist_maxmean_real_y = Hist_maxmean_real[0]*N_maxmean
    Hist_maxmean_real_x = Hist_maxmean_real[1]
    Hist_maxmean_real_x = (Hist_maxmean_real_x[0:bins_W-1]+Hist_maxmean_real_x[1:bins_W])/2
    Hist_maxmean_fake_y = Hist_maxmean_fake[0]*N_maxmean
    Hist_maxmean_fake_x = Hist_maxmean_real_x
    
    List_Hist_r = [Hist_mean_real_y,Hist_skewness_real_y,Hist_CV_real_y,Hist_kurtosis_real_y,Hist_maxmean_real_y]
    List_Hist_f = [Hist_mean_fake_y,Hist_skewness_fake_y,Hist_CV_fake_y,Hist_kurtosis_fake_y,Hist_maxmean_fake_y]
    List_Hist_x = [Hist_mean_fake_x,Hist_skewness_fake_x,Hist_CV_fake_x,Hist_kurtosis_fake_x,Hist_maxmean_fake_x]
    
    std_mean = (np.std(mean_real)+np.std(mean_fake))/2
    Mean_EMD = EMD(mean_real/std_mean,mean_fake/std_mean)
    
    std_skewness = (np.std(skewness_real)+np.std(skewness_fake))/2
    Skewness_EMD = EMD(skewness_real/std_skewness,skewness_fake/std_skewness)

    std_CV = (np.std(CV_real)+np.std(CV_fake))/2
    CV_EMD = EMD(CV_real/std_CV,CV_fake/std_CV)

    std_kurtosis = (np.std(kurtosis_real)+np.std(kurtosis_fake))/2
    Kurtosis_EMD = EMD(kurtosis_real/std_kurtosis,kurtosis_fake/std_kurtosis)

    std_maxmean = (np.std(maxmean_real)+np.std(maxmean_fake))/2
    Maxmean_EMD = EMD(maxmean_real/std_maxmean,maxmean_fake/std_maxmean)

    List_EMD = [Mean_EMD,Skewness_EMD,CV_EMD,Kurtosis_EMD,Maxmean_EMD]
    Avg_Ind_Index = np.mean(List_EMD)
    return List_Hist_r, List_Hist_f, List_Hist_x, List_EMD, Avg_Ind_Index
