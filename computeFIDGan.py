#from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
from trainer.functionnalTrainingClassifier import train, test, adjust_learning_rate
import tqdm
import numpy as np
from models import *
from dataset.BaselineDataset import GanSampleDataset
from dataset.ClassifierDataset import ClassifierDataset
from dataset.GanDataset import GanDataset
from display.functionnalDisplay import display_random_samples
from scipy import linalg
from torchvision import models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# RESNET Path 

netMB = models.resnet18(pretrained=True)
old_weight = netMB.conv1.weight
netMB.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
netMB.conv1.weight = torch.nn.Parameter(old_weight[:,1,:,:].unsqueeze(1))
netMB = netMB.to(device)
netMB.load_state_dict(torch.load("./improved-netMB.pt"))
netMB = torch.nn.Sequential(*list(netMB.children())[:-1])
netMB = torch.nn.DataParallel(netMB)

netMB.eval()


netNoMB = models.resnet18(pretrained=True)
old_weight = netNoMB.conv1.weight
netNoMB.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
netNoMB.conv1.weight = torch.nn.Parameter(old_weight[:,1,:,:].unsqueeze(1))
netNoMB = netNoMB.to(device)
netNoMB.load_state_dict(torch.load("./improved-netNoMB.pt"))
netNoMB = torch.nn.Sequential(*list(netNoMB.children())[:-1])
netNoMB = torch.nn.DataParallel(netNoMB)

netNoMB.eval()
batch_size= 128
reduced_len = 175000
featureSize = 512
num_workers = 24
# DatasetPath
# datasetPath = os.path.join(os.environ.get('SLURM_TMPDIR'),'BaselineCGenPatches.h5')
#
# datasetPath = os.path.join(os.environ.get('SLURM_TMPDIR'),'GanCGenPatches.h5')
# dataset = GanSampleDataset(datasetPath,num_frames=16,reduced_len=reduced_len, key='patchesNoMb')
# dataLoader = torch.utils.data.DataLoader(dataset, num_workers=6, batch_size=128, pin_memory=True,shuffle=False,drop_last=True)
# all_intermediates= torch.zeros(batch_size*len(dataLoader),featureSize)
# with torch.no_grad():
#     for ibatch, batch in tqdm.tqdm(enumerate(dataLoader)):
#         x,_ = batch
#         x = x.to(device)
#
#         intermediate = netNoMB(x).squeeze()
#         all_intermediates[ibatch:ibatch+128] = intermediate.cpu()
# all_intermediates = np.nan_to_num(all_intermediates.numpy(),0)
# mu_sim = np.mean(all_intermediates,axis=0)
# sigma_sim = np.cov(all_intermediates,rowvar=False)


datasetPath = os.path.join(os.environ.get('SLURM_TMPDIR'),'patchesIQ_small_shuffled/')
dataset = GanDataset(datasetPath, 'testMB.h5', label=1,num_frames=16)
dataLoader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True,shuffle=False,
                                         drop_last=True)
all_intermediates= torch.zeros(batch_size*len(dataLoader),featureSize)
with torch.no_grad():
    for ibatch, batch in tqdm.tqdm(enumerate(dataLoader)):
        x,_ = batch
        x = x.to(device)

        intermediate = netNoMB(x).squeeze()
        all_intermediates[ibatch:ibatch+128] = intermediate.cpu()
all_intermediates = np.nan_to_num(all_intermediates.numpy(),0)
mu_sim = np.mean(all_intermediates,axis=0)
sigma_sim = np.cov(all_intermediates,rowvar=False)


datasetPath = os.path.join(os.environ.get('SLURM_TMPDIR'),'GanCGenPatches.h5')
dataset = GanSampleDataset(datasetPath,num_frames=16,reduced_len=reduced_len, key='patchesMb')
dataLoader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True,shuffle=False,
                                         drop_last=True)
all_intermediates= torch.zeros(batch_size*len(dataLoader),featureSize)
with torch.no_grad():
    for ibatch, batch in tqdm.tqdm(enumerate(dataLoader)):
        x,_ = batch
        x = x.to(device)

        intermediate = netNoMB(x).squeeze()
        all_intermediates[ibatch:ibatch+128] = intermediate.cpu()
all_intermediates = np.nan_to_num(all_intermediates.numpy(),0)
mu_real = np.mean(all_intermediates,axis=0)
sigma_real = np.cov(all_intermediates,rowvar=False)
eps= np.eye(sigma_real.shape[0]) * 1e-8
print('FID p_gGan vs P_dMb :')
print(np.mean((mu_real - mu_sim)**2) + np.trace(sigma_real)+np.trace(sigma_sim)-2*np.trace(linalg.sqrtm(((sigma_real+eps)@(sigma_sim+eps)))))


# DatasetPath
dataRealPrefix = os.path.join(os.environ.get(
        'SLURM_TMPDIR'), 'patchesIQ_small_shuffled')
datasetMb = GanDataset(dataRealPrefix, 'testNoMB.h5', 0, num_frames=16,reduced_len=reduced_len)
dataLoader = torch.utils.data.DataLoader(datasetMb, num_workers=6, batch_size=batch_size, pin_memory=True,shuffle=False,
                                         drop_last=True)
all_intermediates= torch.zeros(batch_size*len(dataLoader),featureSize)
with torch.no_grad():
    for ibatch, batch in tqdm.tqdm(enumerate(dataLoader)):
        x,_ = batch
        x = x.to(device)
        intermediate = netNoMB(x).squeeze()
        all_intermediates[ibatch:ibatch+128] = intermediate.cpu()

all_intermediates = np.nan_to_num(all_intermediates.numpy(),0)
mu_sim = np.mean(all_intermediates,axis=0)
sigma_sim = np.cov(all_intermediates,rowvar=False)
eps= np.eye(sigma_sim.shape[0]) * 1e-8

print('FID p_gGan vs P_dNoMb :')
print(np.mean((mu_real - mu_sim)**2) + np.trace(sigma_real)+np.trace(sigma_sim)-2*np.trace(linalg.sqrtm(((sigma_real+eps)@(sigma_sim+eps)))))


# # DatasetPath
# dataRealPrefix = os.path.join(os.environ.get(
#         'SLURM_TMPDIR'), 'patchesIQ_small_shuffled')
# datasetMb = GanDataset(dataRealPrefix, 'testMB.h5', 0, num_frames=16,reduced_len=reduced_len)
# dataLoader = torch.utils.data.DataLoader(datasetMb, num_workers=num_workers, batch_size=batch_size, pin_memory=True,
#                                          shuffle=False,
#                                          drop_last=True)
# all_intermediates= torch.zeros(batch_size*len(dataLoader),featureSize,dtype=float)
# with torch.no_grad():
#     for ibatch, batch in tqdm.tqdm(enumerate(dataLoader)):
#         x,_ = batch
#         x = x.to(device)
#         intermediate = netMB(x).squeeze()
#         all_intermediates[ibatch:ibatch+128] = intermediate.cpu()
#
# all_intermediates = np.nan_to_num(all_intermediates.numpy(),0)
# mu_real = np.mean(all_intermediates,axis=0)
# sigma_real = np.cov(all_intermediates,rowvar=False)
#
#
# print('FID p_gGan vs P_dMb :')
# print(np.mean((mu_real - mu_sim)**2) + np.trace(sigma_real)+np.trace(sigma_sim)-2*np.trace(linalg.sqrtm(((sigma_real+eps)@(sigma_sim+eps)))))
