from comet_ml import Experiment
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import tqdm
from trainCGanUlm import generate_sample
from dataset.GanDataset import  SimuDataset

import h5py
sys.path.append("stylegan3")
from stylegan3.training.networks_stylegan2 import Generator
import numpy as np
from models import *
from dataset.BaselineDataset import BaselineDataset
from display.functionnalDisplay import display_random_samples

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# RESNET Path 
modelPath = 'checkpointGan/65315867_suspicious_flux_8527'
# DatasetPath
SavePath = os.path.join(os.environ.get(
        'SLURM_TMPDIR'),'GanCGenPatches.h5')
nSample = 200000
num_workers=6
batch_size=200
genLatentDim=32
patchSize= (2, 32, 32)
cdim=np.prod( patchSize)
dataSimuPath = os.path.join(os.environ.get(
        'SLURM_TMPDIR'), 'cGenUlmSimu', 'seg1_frames_1000bb_20s_seed_1_0.h5')
   
datasetSimu = SimuDataset(dataSimuPath, num_frames=16)

simuLoader = torch.utils.data.DataLoader(
datasetSimu, num_workers=num_workers//3, batch_size=batch_size, pin_memory=True)

gen = Generator(z_dim=genLatentDim, c_dim=cdim, w_dim=128,
                img_resolution=32, img_channels=2).to(device)
gen.load_state_dict(torch.load(modelPath)['Generator'])
gen.eval()
patchesNoMb = torch.zeros(nSample,*patchSize[1:])
patchesMb = torch.zeros(nSample,*patchSize[1:])
patchesSimu = torch.zeros(nSample,*patchSize[1:])
nSamplePerBatch = np.ceil(nSample/(len(simuLoader) * batch_size)).astype('int')
print('nSamplePerBatch' ,nSamplePerBatch)
with torch.no_grad():
    iPatchStart = 0
    for ibatch,batch_simu in tqdm.tqdm(enumerate(simuLoader)):
        batch_simu = batch_simu.to(device)
        for iSample in range(nSamplePerBatch):
            genNoMb, genMb = generate_sample(gen, batch_simu)
            patchesNoMb[iPatchStart:min(iPatchStart + batch_size,nSample)] = genNoMb.cpu().squeeze(1)
            patchesMb[iPatchStart:min(iPatchStart + batch_size,nSample)] = genMb.cpu().squeeze(1)
            patchesSimu[iPatchStart:min(iPatchStart + batch_size,nSample)] = torch.sqrt(batch_simu[:,0]**2 +batch_simu[:,1]**2 ).cpu()
            iPatchStart += batch_size
            if iPatchStart == nSample:
                break
        if iPatchStart == nSample:
            break

with h5py.File(SavePath,'w') as h5df: 
    h5df.create_dataset('patchesMb',data=patchesMb)
    h5df.create_dataset('patchesNoMb',data=patchesNoMb)
    h5df.create_dataset('patchesSimu',data=patchesSimu)

