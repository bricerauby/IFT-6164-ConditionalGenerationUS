from comet_ml import Experiment
import sys
import os
import numpy as np
import torch
import tqdm
import glob
sys.path.append("stylegan3")
from stylegan3.training.networks_stylegan2 import Discriminator
from stylegan3.training.networks_stylegan2 import Generator
from dataset.GanDataset import GanDataset, SimuDataset
from display.functionnalDisplay import build_figure_samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_sample(generator, batch_simu):
    # reshape simulated sample to conditionned the generator
    batch_size = batch_simu.shape[0]
    genLatentDim = generator.z_dim
    cond = batch_simu.reshape(batch_size, -1).to(device)
    # sample latent space
    z = torch.randn(batch_size, genLatentDim).to(device)
    genNoMb = generator(z, cond)

    # compute the corresponding sample with the MB added (as a complex signal)
    genMb = genNoMb+batch_simu
    genMb = torch.sqrt(genMb[:, 0, :, :]**2 +
                       genMb[:, 1, :, :]**2).unsqueeze(1)
    genNoMb = torch.sqrt(genNoMb[:, 0, :, :]**2 +
                         genNoMb[:, 1, :, :]**2).unsqueeze(1)

    return genNoMb, genMb


def calc_gradient_penalty(discr, real_data, fake_data, lambda_gp=10):
    """
    compute gradient penalty 
    code from : 
    https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
    """
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    discr_interpolates = discr(interpolates, None)

    gradients = torch.autograd.grad(outputs=discr_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(
                                        discr_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def discr_update(real, generated, discr, optim, lambda_gp):
    optim.zero_grad()
    loss = discr(real, None).mean() - discr(generated, None).mean()
    loss.backward()
    gp = calc_gradient_penalty(
        real_data=real, fake_data=generated, discr=discr, lambda_gp=lambda_gp)
    gp.backward()
    optim.step()
    return loss.item()


def train(patch_sizeSimu=(2, 32, 32), nStepsPerEpoch=1000, n_epoch=10,
          num_workers=6, batch_size=64, genLatentDim=32, n_step_discr=3,
          lambda_gp=1e2, lr=1e-4, betas=[0.5, 0.9], experiment=None):
    dataRealPrefix = os.path.join(os.environ.get(
        'SLURM_TMPDIR'), 'patchesIQ_small_shuffled')
    dataSimuPath = os.path.join(os.environ.get(
        'SLURM_TMPDIR'), 'cGenUlmSimu', 'seg1_frames_1000bb_20s_seed_1_0.h5')
    cdim = np.prod(patch_sizeSimu)

    gen = Generator(z_dim=genLatentDim, c_dim=cdim, w_dim=128,
                    img_resolution=32, img_channels=2).to(device)
    discrMB = Discriminator(c_dim=0, img_resolution=32,
                            img_channels=1).to(device)
    discrNoMB = Discriminator(
        c_dim=0, img_resolution=32, img_channels=1).to(device)

    datasetNoMb = GanDataset(dataRealPrefix, 'trainNoMB.h5', 1, num_frames=16)
    datasetMb = GanDataset(dataRealPrefix, 'trainMB.h5', 0, num_frames=16)
    datasetSimu = SimuDataset(dataSimuPath, num_frames=16)

    noMbsampler = torch.utils.data.RandomSampler(
        datasetNoMb, replacement=True, num_samples=int(1e10))
    noMbLoader = torch.utils.data.DataLoader(
        datasetNoMb, num_workers=num_workers//3, batch_size=batch_size, pin_memory=True, sampler=noMbsampler)

    Mbsampler = torch.utils.data.RandomSampler(
        datasetMb, replacement=True, num_samples=int(1e10))
    mbLoader = torch.utils.data.DataLoader(
        datasetMb, num_workers=num_workers//3, batch_size=batch_size, pin_memory=True, sampler=Mbsampler)

    simuSampler = torch.utils.data.RandomSampler(
        datasetSimu, replacement=True, num_samples=int(1e10))
    simuLoader = torch.utils.data.DataLoader(
        datasetSimu, num_workers=num_workers//3, batch_size=batch_size, pin_memory=True, sampler=simuSampler)

    iterNoMB = iter(noMbLoader)
    iterMB = iter(mbLoader)
    iterSimu = iter(simuLoader)

    optimizerGen = torch.optim.Adam(
        gen.parameters(), lr=lr, betas=betas, eps=1e-8)
    optimizerdiscrMb = torch.optim.Adam(
        discrMB.parameters(), lr=lr, betas=betas, eps=1e-8)
    optimizerdiscrNoMb = torch.optim.Adam(
        discrNoMB.parameters(), lr=lr, betas=betas, eps=1e-8)

    for epoch in range(n_epoch):
        print('EPOCH : {}/{}'.format(epoch, n_epoch))
        lossNoMbTotal = 0
        lossMbTotal=0
        lossGenTotal = 0
        for iStep in tqdm.tqdm(range(nStepsPerEpoch)):
            for i in range(n_step_discr):
                gen.eval()
                with torch.no_grad():
                    batch_simu = next(iterSimu).to(device)
                    genNoMb, genMb = generate_sample(gen, batch_simu)

                realNoMb, _ = next(iterNoMB)
                realMb, _ = next(iterMB)
                realNoMb = realNoMb.to(device)
                realMb = realMb.to(device)
                lossMb = discr_update(realMb,
                                      genMb, discrMB,
                                      optimizerdiscrMb,
                                      lambda_gp)
                lossNoMb = discr_update(realNoMb,
                                        genNoMb,
                                        discrNoMB,
                                        optimizerdiscrNoMb,
                                        lambda_gp)
                lossNoMbTotal += lossNoMb
                lossMbTotal += lossMb

            gen.train()
            optimizerGen.zero_grad()
            genNoMb, genMb = generate_sample(gen, batch_simu)
            lossGen = discrMB(genMb, None).mean() + \
                discrNoMB(genNoMb, None).mean()
            lossGen.backward()
            optimizerGen.step()
            lossGenTotal += lossGen.item()
        lossGenTotal /= nStepsPerEpoch 
        lossMbTotal /= n_step_discr * nStepsPerEpoch
        lossNoMbTotal /= n_step_discr * nStepsPerEpoch

        if experiment: 
            experiment.log_metric("loss generator", lossGenTotal, epoch=epoch)
            experiment.log_metric("loss discriminator MB", lossMbTotal, epoch=epoch)
            experiment.log_metric("loss discriminator noMB", lossNoMbTotal, epoch=epoch)

            with torch.no_grad():
                build_figure_samples((torch.sqrt(batch_simu[:,0,:,:]**2 + batch_simu[:,1,:,:]**2).cpu()),
                                     experiment_kwargs={"experiment":experiment, "figure_name":"simulation cond"},dispLabel=False)
                build_figure_samples(genNoMb.cpu(),
                                     experiment_kwargs={"experiment":experiment, "figure_name":"generated samples without MB"},dispLabel=False)
                build_figure_samples(genMb.cpu(),
                                     experiment_kwargs={"experiment":experiment, "figure_name":"generated samples with MB"},dispLabel=False)
        
        print("loss generator {:1.1e} | loss discriminator MB {:1.1e} | loss discriminator No MB {:1.1e}".format(lossGenTotal,lossMbTotal,lossNoMbTotal))
        state = {
            'DiscrNoMB': discrNoMB.state_dict(),
            'DiscrMB': discrMB.state_dict(),
            'Generator': gen.state_dict(),
        }
        if not os.path.isdir('checkpointGan'):
            os.mkdir('checkpointGan')
        torch.save(state, './checkpointGan/' + experiment.name)
        print('Model Saved!')



if __name__ == '__main__':

    parameters = {"patch_sizeSimu": (2, 32, 32),
                  "nStepsPerEpoch": 100,
                  "n_epoch": 100,
                  "num_workers": 6,
                  "batch_size": 64,
                  "genLatentDim": 32,
                  "n_step_discr": 5,
                  "lambda_gp": 1e2,
                  "lr": 5e-5,
                  "betas": [0.5, 0.9],
                  }
    experiment = Experiment(project_name='cgenulm',
                            workspace='bricerauby', auto_log_co2=False)
    experiment.add_tag('wgan-gp')
    experiment.set_name(os.environ.get('SLURM_JOB_ID') +
                        '_' + experiment.get_name())
    code_list = glob.glob("**/*.py", recursive=True)
    experiment.log_parameters(parameters)
    train(**parameters, experiment=experiment)
