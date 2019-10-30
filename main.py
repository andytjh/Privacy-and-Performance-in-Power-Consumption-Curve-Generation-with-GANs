#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, grad
from tensorboardX import SummaryWriter
from QualityMetrics import Indicators
from MakeHist import MakeHist
from matplotlib import pyplot as plt
import importlib

##
## Command line parameters ##
##

parser = argparse.ArgumentParser(description='Train the selected model')

parser.add_argument('--run',type=int,help='Number of the run')
parser.add_argument('--model_ver',type=str,help='Version of the model. Must be the name of the module containing the desired model.')
parser.add_argument('--eta',type=float,default=0.12,help='Weight of the sum between the losses of the two discriminators (Only relevant for GANConv1Dv1WIa.py model).')
parser.add_argument('--alpha',type=float,default=300,help='Alpha parameter for pre-processing.')
parser.add_argument('--grad_norm_reg',type=bool,default=False,help='If gradient-norm regularization is applied.')
parser.add_argument('--gamma',type=float,default=0.01,help='Rate for gradient-norm regularization.')
parser.add_argument('--n_epochs',type=int,default=150,help='Number of epochs for training.')
parser.add_argument('--batch_size',type=int,default=20,help='Batch size.')
parser.add_argument('--lr_g',type=float,default=0.0001,help='Learning rate for the generator.')
parser.add_argument('--lr_d',type=float,default=0.00001,help='Learning rate for the discriminator.')
parser.add_argument('--n_critic',type=int,default=5,help='Number of discriminator steps per generator step.')

opt = parser.parse_args()

model_ver = opt.model_ver
run = opt.run

# Directory for saving TensorBoard files, numpy arrays contaning the results of the attacks and the weights of trained models.

saving_dir = './Logs/'+model_ver

# Instantiate Tensorboar SummaryWriter

writer = SummaryWriter(saving_dir+'/Tensorboard/exp'+str(run))

##
## Creating Random list of clients ##
##

# First 'step' elements of the list will be selected as training data

data_dir = "./DataSets"
dir_list = os.listdir(data_dir)
random.shuffle(dir_list)
step = int(len(dir_list)/10.0)

# Saving the training set suffle

list_directory = saving_dir+'/npdata/dirlist'
if not os.path.exists(list_directory):
    os.makedirs(list_directory)
np.save(list_directory+'/List'+str(run)+'.npy',dir_list)

# Arranging clients into subsets. The first subset will be the training set.

subset_list = []
universe = np.empty(shape=[0,336], dtype='float32')
for i in range(0,len(dir_list),step):
    np_aux = np.empty(shape=[0,336], dtype='float32')
    if ((len(dir_list)-i)>=step):
        for j in range(step):
            aux = np.load(data_dir+'/'+dir_list[i+j])
            np_aux = np.append(np_aux,aux,axis=0)
            universe = np.append(universe,aux,axis=0)
        subset_list.append(np_aux)

# Saving alpha and maximum for transformation and inverse
# Maximum taken over the universe
    
alpha = opt.alpha
train_aux = np.arcsinh(universe*alpha)/alpha
save_max = np.reshape(train_aux,-1).max()

##
## Set-up ##
##

# Checking for cuda

if torch.cuda.is_available():
    cuda = True
else: 
    cuda = False
    
# Loss Function

loss = nn.BCEWithLogitsLoss() #Note that this lost function integrates the softmax activation function for numerical stability.

# Instantiating generator and discriminator models

module_arch = importlib.__import__(model_ver)

generator = module_arch.GeneratorConv1D()
if model_ver == 'GANConv1Dv1WIa':
    discriminator = module_arch.DiscriminatorSignal()
    discriminator_I = module_arch.DiscriminatorIndicators()
else:
    discriminator = module_arch.DiscriminatorConv1D()

if cuda:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()
    if model_ver == 'GANConv1Dv1WIa':
        discriminator_I.cuda()
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Defining pre-processing transformation and inverse transformation

# Works with numpy arrays!!!
def transformation(array,alpha,save_max):
    array = np.arcsinh(array*alpha)/alpha
    array = (array*2.0)/save_max - 1.0
    array = array[:,np.newaxis,:]
    return array

# Works with pytorch tensors!!!
def inverse_trans(arrtensor,alpha,save_max):
    arrtensor = (arrtensor+1.0)*save_max/2.0
    return torch.sinh(arrtensor*alpha)/alpha

# Optimizer

optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr_g)
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr_d)
if model_ver =='GANConv1Dv1WIa':
    optimizer_D_I = torch.optim.Adam(discriminator_I.parameters(),lr=opt.lr_d)

# Loading training set

training_set = subset_list[0]

class TimeSeriesLCL(Dataset):
    def __init__(self, npy_array,alpha,save_max):
        self.x_train = npy_array
        self.x_train = np.arcsinh(self.x_train*alpha)/alpha
        self.x_train = (self.x_train*2.0)/save_max - 1.0
        self.x_train = self.x_train[:,np.newaxis,:]
    
    def __len__(self):
        return self.x_train.shape[0]
    
    def __getitem__(self, idx):
        example = self.x_train[idx,]
        return example
    
x_train = TimeSeriesLCL(training_set,alpha,save_max)

# Some parameters for training

if model_ver == 'GANConv1Dv0':
    latent_space_dim = 25
else:
    latent_space_dim = 42

eta = opt.eta
gamma = opt.gamma
n_epochs = opt.n_epochs
batch_size = opt.batch_size
steps_generator = opt.n_critic
steps_discriminator = 1

dataloader = DataLoader(x_train,batch_size=batch_size,shuffle=True)

generated_samples = []
real_examples = []

##
## Training ##
##

for epoch in range(n_epochs):
    
    for i, example_batch in enumerate(dataloader):
        
#       Ground truths for the discriminator
        valid = Variable(Tensor(example_batch.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(example_batch.shape[0], 1).fill_(0.0), requires_grad=False)
        
#       Configuring input
        example_batch = example_batch.type(Tensor)
        real_examples.append(torch.squeeze(example_batch))
        
#       Generating samples
        z = Tensor(np.random.normal(size=[example_batch.shape[0],latent_space_dim]))
        
        generated_sample = generator(z)
        generated_samples.append(torch.squeeze(generated_sample.detach()))
        
        if model_ver =='GANConv1Dv1WIa':
#           Train generator
            if i%steps_generator == 0:
                optimizer_G.zero_grad()
                g_loss_S = loss(discriminator(generated_sample),valid)
                g_loss_I = loss(discriminator_I(generated_sample),valid)
                basic_g_loss = (1.0-eta)*g_loss_S + eta*g_loss_I
                basic_g_loss.backward()
                optimizer_G.step()

#           Train Discriminator
            if i%steps_discriminator == 0:
                optimizer_D.zero_grad()
                real_loss = loss(discriminator(example_batch),valid)
                fake_loss = loss(discriminator(generated_sample.detach()),fake)           
                if opt.grad_norm_reg:
                    basic_d_loss = (real_loss + fake_loss)/2.0
                    d_grad = grad(basic_d_loss,discriminator.parameters(),create_graph=True)
                    dn2 = torch.sqrt(sum([grd.norm()**2 for grd in d_grad]))
                    final_d_loss = basic_d_loss - gamma*dn2
                else:
                    final_d_loss = (real_loss + fake_loss)/2.0
                final_d_loss.backward()
                optimizer_D.step()

                optimizer_D_I.zero_grad()
                real_loss_I = loss(discriminator_I(example_batch),valid)
                fake_loss_I = loss(discriminator_I(generated_sample.detach()),fake)
                d_loss_I = (real_loss_I + fake_loss_I)/2.0
                d_loss_I.backward()
                optimizer_D_I.step()
        else:
#           Train generator
            if i%steps_generator == 0:
                optimizer_G.zero_grad()
                basic_g_loss = loss(discriminator(generated_sample),valid)
                basic_g_loss.backward()
                optimizer_G.step()

#           Train Discriminator
            if i%steps_discriminator == 0:
                optimizer_D.zero_grad()
                real_loss = loss(discriminator(example_batch),valid)
                fake_loss = loss(discriminator(generated_sample.detach()),fake)           
                if opt.grad_norm_reg:
                    basic_d_loss = (real_loss + fake_loss)/2.0
                    d_grad = grad(basic_d_loss,discriminator.parameters(),create_graph=True)
                    dn2 = torch.sqrt(sum([grd.norm()**2 for grd in d_grad]))
                    final_d_loss = basic_d_loss - gamma*dn2
                else:
                    final_d_loss = (real_loss + fake_loss)/2.0
                final_d_loss.backward()
                optimizer_D.step()
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch+1, n_epochs, i+1, len(dataloader), final_d_loss.item(), basic_g_loss.item())
        )
        
        # Saving the loss for the Generator and Discriminator
        
        writer.add_scalar('Generator loss', basic_g_loss.item(), 1+i+(epoch*len(dataloader)))
        writer.add_scalar('Discriminator loss', final_d_loss.item(),1+i+(epoch*len(dataloader)))
        
        # Plotting artificially generated samples, empirical distributions of the indicators and saving plots to tensorboard.
        
        if (((i+1)*batch_size) % 800) == 0:
            
            generated_samples = torch.cat(generated_samples)
            generated_samples = inverse_trans(generated_samples,alpha,save_max)
            indicators_gen = Indicators(generated_samples)
            indicators_gen = Tensor.cpu(indicators_gen)
            indicators_gen = indicators_gen.data.numpy()
            
            real_examples = torch.cat(real_examples)
            real_examples = inverse_trans(real_examples,alpha,save_max)
            indicators_real = Indicators(real_examples)
            indicators_real = Tensor.cpu(indicators_real)
            indicators_real = indicators_real.data.numpy()
            
            g_sample = generated_samples[0,:]
            g_sample = Tensor.cpu(g_sample)
            g_sample = g_sample.data.numpy()
            g_sample_fig = plt.figure(0)
            plt.plot(g_sample)
            plt.title('Generated Sample')
            plt.ylabel('Energy (KWh)')
            plt.xlabel('Time (half hour)')
            writer.add_figure('Generated Sample', g_sample_fig,1+i+(epoch*len(dataloader)))
            
            List_Hist_r, List_Hist_f, List_Hist_x, List_EMD, Avg_Ind_Index = MakeHist(indicators_real,indicators_gen)
            
            mean_Hist = plt.figure(0)
            plt.plot(List_Hist_x[0],List_Hist_r[0])
            plt.plot(List_Hist_x[0],List_Hist_f[0])
            plt.legend(['Real','Fake'])
            plt.title('Empirical distribution of the mean.')
            plt.xlabel('Mean')
            writer.add_scalar('EMD of the mean', List_EMD[0],1+i+(epoch*len(dataloader)))
            writer.add_figure('Histogram of the mean', mean_Hist,1+i+(epoch*len(dataloader)))
                              
            skewness_Hist = plt.figure(1)
            plt.plot(List_Hist_x[1],List_Hist_r[1])
            plt.plot(List_Hist_x[1],List_Hist_f[1])
            plt.legend(['Real','Fake'])
            plt.title('Empirical distribution of the skewness.')
            plt.xlabel('Skewness')
            writer.add_scalar('EMD of the skewness', List_EMD[1],1+i+(epoch*len(dataloader)))
            writer.add_figure('Histogram of the skewness', skewness_Hist,1+i+(epoch*len(dataloader)))
                              
            CV_Hist = plt.figure(2)
            plt.plot(List_Hist_x[2],List_Hist_r[2])
            plt.plot(List_Hist_x[2],List_Hist_f[2])
            plt.legend(['Real','Fake'])
            plt.title('Empirical distribution of the CV.')
            plt.xlabel('Coefficient of variation')
            writer.add_scalar('EMD of the CV', List_EMD[2],1+i+(epoch*len(dataloader)))
            writer.add_figure('Histogram of the CV', CV_Hist,1+i+(epoch*len(dataloader)))
                              
            kurtosis_Hist = plt.figure(3)
            plt.plot(List_Hist_x[3],List_Hist_r[3])
            plt.plot(List_Hist_x[3],List_Hist_f[3])
            plt.legend(['Real','Fake'])
            plt.title('Empirical distribution of the kurtosis.')
            plt.xlabel('Kurtosis')
            writer.add_scalar('EMD of the kurtosis', List_EMD[3],1+i+(epoch*len(dataloader)))
            writer.add_figure('Histogram of the kurtosis', kurtosis_Hist,1+i+(epoch*len(dataloader)))
                              
            maxmean_Hist = plt.figure(4)
            plt.plot(List_Hist_x[4],List_Hist_r[4])
            plt.plot(List_Hist_x[4],List_Hist_f[4])
            plt.legend(['Real','Fake'])
            plt.title('Empirical distribution of the max-mean ratio.')
            plt.xlabel('Max-mean ratio')
            writer.add_scalar('EMD of the max-mean ratio', List_EMD[4],1+i+(epoch*len(dataloader)))
            writer.add_figure('Histogram of the max-mean ratio', maxmean_Hist,1+i+(epoch*len(dataloader)))
                              
            writer.add_scalar('Average Indicator Index', Avg_Ind_Index,1+i+(epoch*len(dataloader)))
                        
            generated_samples = []
            real_examples = []

# Saving the model

mod_directory = saving_dir+'/Trained'
if not os.path.exists(mod_directory):
    os.makedirs(mod_directory)
torch.save(generator.state_dict(), mod_directory+'/GEN_run'+str(run)+'.pth')
torch.save(discriminator.state_dict(), mod_directory+'/DIS_run'+str(run)+'.pth')
print('Model Saved')

##
## Gradient Norm Attack ##
##

batch_size = 1 # batch size for the attack

generator.eval()
discriminator.eval()

# The attack itself
norms_per_subset = []
scores_per_subset = []

for i in range(10): 
    
    norm = []
    scores = []

    for j in range(step): 
        
        examples = np.load(data_dir+'/'+dir_list[(i*step)+j])
        examples = transformation(examples,alpha,save_max)

        client_norm = np.empty([0])
        client_score = np.empty([0])

        for k in range(0,examples.shape[0],batch_size):

        #   Configuring Input
            example_batch = Tensor(examples[k:k+batch_size,:])

        #   Ground truth for the discriminator

            valid = Variable(Tensor(example_batch.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(example_batch.size(0), 1).fill_(0.0), requires_grad=False)

        #   Generating fake samples

            z = Tensor(np.random.normal(size=[example_batch.size(0),latent_space_dim]))
            generated = generator(z)

        #   Taking the gradient of the discriminator 

            valid_loss = loss(discriminator(example_batch),valid)
            fake_loss = loss(discriminator(generated.detach()),fake)

            total_loss = (valid_loss + fake_loss)/2.0
            discriminator.zero_grad()

#             total_loss.backward(retain_graph=True)
            
        #   Saving discriminator score for sample
        
            score = discriminator(example_batch)
            score = Tensor.cpu(score)
            score = score.data.numpy()
            client_score = np.append(client_score, score)

        #   Calculating the norm
                            
            d_grad = grad(total_loss,discriminator.parameters(),create_graph=True)
            dn2 = torch.sqrt(sum([grd.norm()**2 for grd in d_grad]))
            dn2 = dn2.detach()
            dn2 = Tensor.cpu(dn2)
            dn2 = dn2.data.numpy()
            client_norm = np.append(client_norm,dn2)

    #   Saving the norm for a client
    
        scores.append(client_score)
        norm.append(client_norm)
        # Loop through clients
        
    norms_per_subset.append(norm)
    scores_per_subset.append(scores)
    # Loop through subsets
    
norms_directory = saving_dir+'/npdata/Norms'
if not os.path.exists(norms_directory):
    os.makedirs(norms_directory)
np.save(norms_directory+'/SSNorm'+str(run)+'.npy',norms_per_subset)

scores_directory = saving_dir+'/npdata/Scores'
if not os.path.exists(scores_directory):
    os.makedirs(scores_directory)
np.save(scores_directory+'/SSScore'+str(run)+'.npy',scores_per_subset)

## 
## Classification ##
##

# Using the Norm

mean_norms_per_client = []
mean_norms_per_subset = []
std_norms_per_client = []
std_norms_per_subset = []

# Going through norms for all samples.
# Saving per client and per subset mean and std.

for i in range(len(norms_per_subset)):
    norms_per_client = norms_per_subset[i]
    mean_norm_client_for_subset = []
    std_norm_client_for_subset = []
    all_norms_subset = np.empty([0])
    for j in range(step):
        client_norms = norms_per_client[j]
        all_norms_subset = np.append(all_norms_subset,client_norms)
        mean_client_norm = np.mean(client_norms)
        std_client_norm = np.std(client_norms)
        mean_norm_client_for_subset.append(mean_client_norm)
        std_norm_client_for_subset.append(std_client_norm)
    mean_norms_per_client.append(mean_norm_client_for_subset)
    mean_norms_per_subset.append(np.mean(mean_norm_client_for_subset))
    std_norms_per_client.append(std_norm_client_for_subset)
    std_norms_per_subset.append(np.std(all_norms_subset))


# Classifying Subset Based
    
subset_ranking_mean = np.argsort(mean_norms_per_subset)
subset_ranking_std = np.argsort(std_norms_per_subset)

ranksSS_mean_directory = saving_dir+'/npdata/RankMeanPSS/'
if not os.path.exists(ranksSS_mean_directory):
    os.makedirs(ranksSS_mean_directory)
np.save(ranksSS_mean_directory+'RankpSubset'+str(run)+'.npy',subset_ranking_mean)

ranksSS_std_directory = saving_dir+'/npdata/RankStdPSS/'
if not os.path.exists(ranksSS_std_directory):
    os.makedirs(ranksSS_std_directory)
np.save(ranksSS_std_directory+'RankpSubset'+str(run)+'.npy',subset_ranking_std)
    
# Classifying Client Based

mean_arb_client_ranking = []
std_arb_client_ranking = []
for j in range(100):
    rand_select_norms = []
    rand_select_norms_std = []
    for i in range(len(mean_norms_per_client)):
        selected_client = np.random.choice(step)
        norm_of_client = mean_norms_per_client[i][selected_client]
        std_of_client = std_norms_per_client[i][selected_client]
        rand_select_norms.append(norm_of_client)
        rand_select_norms_std.append(std_of_client)
    aux = np.argsort(rand_select_norms)
    aux_1 = np.argsort(rand_select_norms_std)
    mean_arb_client_ranking.append(aux)
    std_arb_client_ranking.append(aux_1)

ranksC_mean_directory = saving_dir+'/npdata/RankMeanPC'
if not os.path.exists(ranksC_mean_directory):
    os.makedirs(ranksC_mean_directory)
np.save(ranksC_mean_directory+'/RankpClient'+str(run)+'.npy',mean_arb_client_ranking)

ranksC_std_directory = saving_dir+'/npdata/RankStdPC'
if not os.path.exists(ranksC_std_directory):
    os.makedirs(ranksC_std_directory)
np.save(ranksC_std_directory+'/RankpClient'+str(run)+'.npy',std_arb_client_ranking)

# Using the scores

mean_score_per_client = []
mean_score_per_subset = []

for i in range(len(scores_per_subset)):
    scores_per_client = scores_per_subset[i]
    mean_scores_client_for_subset = []
    for j in range(step):
        client_scores = scores_per_client[j]
        mean_client_score = np.mean(client_scores)
        mean_scores_client_for_subset.append(mean_client_score)
    mean_score_per_client.append(mean_scores_client_for_subset)
    mean_score_per_subset.append(np.mean(mean_scores_client_for_subset))

# Classifying Subset Based
    
subset_ranking_score = np.argsort(mean_score_per_subset)

ranksSS_score_directory = saving_dir+'/npdata/RankMeanScorePSS/'
if not os.path.exists(ranksSS_score_directory):
    os.makedirs(ranksSS_score_directory)
np.save(ranksSS_score_directory+'RankpSubset'+str(run)+'.npy',subset_ranking_score)

# Classifying Client Based

score_arb_client_ranking = []
for j in range(100):
    rand_select_scores = []
    for i in range(len(mean_score_per_client)):
        selected_client = np.random.choice(step)
        score_of_client = mean_score_per_client[i][selected_client]
        rand_select_scores.append(score_of_client)
    aux = np.argsort(rand_select_scores)
    score_arb_client_ranking.append(aux)

ranksC_score_directory = saving_dir+'/npdata/RankMeanScorePC'
if not os.path.exists(ranksC_score_directory):
    os.makedirs(ranksC_score_directory)
np.save(ranksC_score_directory+'/RankpClient'+str(run)+'.npy',score_arb_client_ranking)

# Using the Indicators

AII_list = []

for i in range(10): 
    examples = np.empty([0,336])
    generated_list = []
    for j in range(step): 
        aux = np.load(data_dir+'/'+dir_list[(i*step)+j])
        num_samples = aux.shape[0]
        z = Tensor(np.random.normal(size=[num_samples,latent_space_dim]))
        generated = generator(z)
        if num_samples == 1:
            aux_aux = torch.squeeze(generated.detach())
            generated_list.append(torch.unsqueeze(aux_aux,0))
        else:
            generated_list.append(torch.squeeze(generated.detach()))
        examples = np.append(examples,aux,axis=0)
        
    examples = Tensor(examples)
    generated_s = torch.cat(generated_list)
    generated = inverse_trans(generated_s,alpha,save_max)
    
    indicators_gen = Indicators(generated)
    indicators_gen = Tensor.cpu(indicators_gen)
    indicators_gen = indicators_gen.data.numpy()
    
    indicators_real = Indicators(examples)
    indicators_real = Tensor.cpu(indicators_real)
    indicators_real = indicators_real.data.numpy()
    
    Avg_Ind_Index = MakeHist(indicators_real,indicators_gen)[4]
    
    AII_list.append(Avg_Ind_Index)
    
indicator_proximity_ranking = np.argsort(AII_list)

ranksSS_Indicators_directory = saving_dir+'/npdata/RankIndicatorsPSS/'
if not os.path.exists(ranksSS_Indicators_directory):
    os.makedirs(ranksSS_Indicators_directory)
np.save(ranksSS_Indicators_directory+'RankpSubset'+str(run)+'.npy', indicator_proximity_ranking)
