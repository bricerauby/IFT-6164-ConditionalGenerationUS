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
from models.gan import Unet
from display.functionnalDisplay import build_figure_samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_sample(generator, batch_simu):
    # reshape simulated sample to conditionned the generator
    batch_size = batch_simu.shape[0]
    genLatentDim = generator.z_dim
    cond = batch_simu.reshape(batch_size, -1).to(device)
    # sample latent space
    z = torch.randn(batch_size, genLatentDim).to(device)
    genMb = generator(z, cond)
    return genMb



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

    gen_sim2real = Generator(z_dim=genLatentDim, c_dim=cdim, w_dim=128,
                    img_resolution=32, img_channels=1).to(device)
    gen_real2sim = Unet(input_channel=1, n_chans_out=1, nblocs=3,dim=2).to(device)
    discr_f = Discriminator(c_dim=0, img_resolution=32,
                            img_channels=2).to(device)

    datasetMb = GanDataset(dataRealPrefix, 'trainMB.h5', 1, num_frames=16)
    datasetSimu = SimuDataset(dataSimuPath, num_frames=16, real=True)

    Mbsampler = torch.utils.data.RandomSampler(
        datasetMb, replacement=True, num_samples=int(1e10))
    mbLoader = torch.utils.data.DataLoader(
        datasetMb, num_workers=num_workers//3, batch_size=batch_size, pin_memory=True, sampler=Mbsampler)

    simuSampler = torch.utils.data.RandomSampler(
        datasetSimu, replacement=True, num_samples=int(1e10))
    simuLoader = torch.utils.data.DataLoader(
        datasetSimu, num_workers=num_workers//3, batch_size=batch_size, pin_memory=True, sampler=simuSampler)

    iterMB = iter(mbLoader)
    iterSimu = iter(simuLoader)

    optimizerGenSim2Real = torch.optim.Adam(
        gen_sim2real.parameters(), lr=lr, betas=betas, eps=1e-8)
    optimizerGenReal2Sim = torch.optim.Adam(
        gen_real2sim.parameters(), lr=lr, betas=betas, eps=1e-8)
    optimizerdiscr_f = torch.optim.Adam(
        discr_f.parameters(), lr=lr, betas=betas, eps=1e-8)


    for epoch in range(n_epoch):
        print('EPOCH : {}/{}'.format(epoch, n_epoch))
        loss_discrTotal = 0
        lossGenSim2RealTotal = 0
        lossGenReal2SimTotal = 0
        lossCycleTotal=0
        for iStep in tqdm.tqdm(range(nStepsPerEpoch)):
            for i in range(n_step_discr):
                gen_sim2real.eval()
                with torch.no_grad():
                    batch_simu = next(iterSimu).to(device)
                    genMb = generate_sample(gen_sim2real, batch_simu)
                with torch.no_grad():
                    batch_real, _ = next(iterMB)
                    batch_real = batch_real.to(device)
                    genSimu = gen_real2sim(batch_real)

                batch_real, _ = next(iterMB)
                batch_real = batch_real.to(device)
                realMb = torch.cat([batch_real,genSimu],dim=1)
                genMb = torch.cat([genMb,batch_simu],dim=1)

                loss_discr = discr_update(realMb,
                                      genMb, discr_f,
                                      optimizerdiscr_f,
                                      lambda_gp)
                loss_discrTotal += loss_discr

            gen_sim2real.train()
            gen_real2sim.train()
            optimizerGenReal2Sim.zero_grad()
            optimizerGenSim2Real.zero_grad()

            batch_simu = next(iterSimu).to(device)
            batch_real, _ = next(iterMB)
            batch_real = batch_real.to(device)

            genMb = generate_sample(gen_sim2real, batch_simu)
            genSimu = gen_real2sim(batch_real)

            batch_simu_hat = gen_real2sim(genMb)
            lossCycle = torch.mean((batch_simu_hat - batch_simu)**2)
            realMb = torch.cat([batch_real, genSimu],dim=1)
            genMb = torch.cat([genMb, batch_simu],dim=1)

            lossGenSim2Real = discr_f(genMb, None).mean() 
            lossGenReal2sim = - discr_f(realMb, None).mean()
            loss = lossCycle + lossGenReal2sim + lossGenSim2Real
            loss.backward()
            optimizerGenReal2Sim.step()
            optimizerGenSim2Real.step()
            lossGenSim2RealTotal += lossGenSim2Real.item()
            lossGenReal2SimTotal += lossGenReal2sim.item()
            lossCycleTotal += lossCycle.item()

        lossGenSim2RealTotal /= nStepsPerEpoch 
        lossGenReal2SimTotal /= nStepsPerEpoch 
        loss_discrTotal /= n_step_discr * nStepsPerEpoch
        lossCycleTotal /= nStepsPerEpoch 
        if experiment is not None: 
            experiment.log_metric("loss generator Sim2Real", lossGenSim2RealTotal, epoch=epoch)
            experiment.log_metric("loss generator Real2Sim", lossGenReal2SimTotal, epoch=epoch)
            experiment.log_metric("loss discriminator", loss_discrTotal, epoch=epoch)
            experiment.log_metric("loss cycle", lossCycleTotal, epoch=epoch)

            with torch.no_grad():
                build_figure_samples(batch_simu.cpu(),
                                     experiment_kwargs={"experiment":experiment, "figure_name":"simulation cond"},dispLabel=False)
                build_figure_samples(batch_real.cpu(),
                        experiment_kwargs={"experiment":experiment, "figure_name":"Real data sample"},dispLabel=False)
                build_figure_samples(genSimu.cpu(),
                                     experiment_kwargs={"experiment":experiment, "figure_name":"Estimated denoised from real"},dispLabel=False)
                build_figure_samples(genMb.cpu(),
                                     experiment_kwargs={"experiment":experiment, "figure_name":"generated samples with MB"},dispLabel=False)
        
        print("loss generator {:1.1e} | loss discriminator {:1.1e}".format(lossGenSim2RealTotal,loss_discrTotal))
        state = {
            'discr_f': discr_f.state_dict(),
            'gen_sim2real': gen_sim2real.state_dict(),
            'gen_real2sim': gen_real2sim.state_dict(),
        }
        if not os.path.isdir('checkpointGan'):
            os.mkdir('checkpointGan')
        if experiment is not None: 
            torch.save(state, './checkpointGan/' + experiment.name)
        else : 
            torch.save(state, './checkpointGan/' + 'model_state')
        print('Model Saved!')



if __name__ == '__main__':

    parameters = {"patch_sizeSimu": (1, 32, 32),
                  "nStepsPerEpoch": 100,
                  "n_epoch": 100,
                  "num_workers": 6,
                  "batch_size": 32,
                  "genLatentDim": 32,
                 "n_step_discr": 5,
                  "lambda_gp": 1e2,
                  "lr": 5e-5,
                  "betas": [0.5, 0.9],
                  }
    experiment = Experiment(project_name='cgenulm',
                            workspace='bricerauby', auto_log_co2=False)
    experiment.add_tag('wgan-gp-style-conditionned-cycle')
    experiment.set_name(os.environ.get('SLURM_JOB_ID') +
                        '_' + experiment.get_name())
    code_list = glob.glob("**/*.py", recursive=True)
    for code in code_list:
        experiment.log_code(file_name=code)
    experiment.log_parameters(parameters)
    train(**parameters, experiment=experiment)
