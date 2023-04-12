# ------python import------------------------------------
import os

import warnings


import torch
import torch.distributed as dist
import logging

import torchvision.datasets
import wandb
from torch.optim.swa_utils import AveragedModel

# -----local imports---------------------------------------
from PytorchTemplate.Parser import init_parser
from PytorchTemplate.variation.GanExperiment import GanExperiment as Experiment
from PytorchTemplate import names



# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def main() :



    parser = init_parser()
    args = parser.parse_args()
    config = vars(args)
    # -------- proxy config ----------------------------------------
    #
    # proxy = urllib.request.ProxyHandler(
    #     {
    #         "https": "",
    #         "http": "",
    #     }
    # )
    # os.environ["HTTPS_PROXY"] = ""
    # os.environ["HTTP_PROXY"] = ""
    # # construct a new opener using your proxy settings
    # opener = urllib.request.build_opener(proxy)
    # # install the openen on the module-level
    # urllib.request.install_opener(opener)






    #----------- load the datasets--------------------------------
    batch_size = config["batch_size"]
    shuffle = True
    num_workers = config["num_worker"]
    pin_memory = True
    from torchvision import transforms,datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([

        # you can add here data augmentation with prob in config["augment_prob"]
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]))

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([

        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]))
    #pytorch random sampler
    num_samples  = 200 if config["debug"] else len(train_dataset)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=num_samples , generator=None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )







    #------------ Training --------------------------------------



    # setting up for the  experiment

   
    from PytorchTemplate.models.StyleGAN3 import Generator, Discriminator
    generator = Generator(z_dim=32, c_dim=0, w_dim=128, img_resolution=config["image_size"], img_channels=3)
    discriminator = Discriminator(c_dim=0, img_resolution=config["image_size"], img_channels=3)

    if torch.__version__>"2.0" and not config["debug"] and False :
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)
    experiment = Experiment(names, config)
    experiment.compile(
        generator=generator,
        discriminator=discriminator,
        optimizer="AdamW",
        criterion="Dice",
        train_loader=train_loader,
        val_loader=val_loader,
    )



    results = experiment.train()

    experiment.end()




if __name__ == "__main__":
  main()
