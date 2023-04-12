# ------python import------------------------------------
import os

import warnings

import timm
import torch
import torch.distributed as dist
import logging

import torchvision.datasets
import wandb
from torch.optim.swa_utils import AveragedModel

# -----local imports---------------------------------------
from PytorchTemplate.Parser import init_parser
from PytorchTemplate.Experiment import Experiment as Experiment
from PytorchTemplate import names
from PytorchTemplate.Dataset import Dataset


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
    from PytorchTemplate.Dataset import CIFAR10Im2Im
    self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([

        transforms.resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ]))

    self.train_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([

        transforms.resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )







    #------------ Training --------------------------------------



    # setting up for the  experiment

    experiment = Experiment(names, config)
    experiment.compile(
        model_name=config["model"],
        optimizer = "AdamW",
        criterion="Dice",
        train_loader=train_loader,
        val_loader=val_loader,
        final_activation="softmax",
    )



    results = experiment.train()

    experiment.end()




if __name__ == "__main__":
  main()
