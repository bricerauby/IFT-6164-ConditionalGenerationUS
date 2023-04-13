import numpy as np
import torch
import torch.distributed as dist
import tqdm

from torchvision import transforms
from torch import autograd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""


import tqdm

import torch
import inspect
import logging
from PytorchTemplate.Experiment import Experiment
from PytorchTemplate.Metrics import Metrics

import torchmetrics


def compute_gp(netD, real_data, fake_data,labels):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = eps * real_data + (1 - eps) * fake_data

    # get logits for interpolated images
    interp_logits = netD(interpolation,labels)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


class GanExperiment(Experiment):

    def watch(self):
        if self.rank == 0 and self.tracker is not None:
            self.tracker.watch(self.generator)
            self.tracker.watch(self.discriminator)

    
    def save_weights(self):
        if self.rank == 0 :
            if dist.is_initialized() :
                #discriminator
                torch.save(self.discriminator.module.state_dict(), f"{self.weight_dir}/{self.discriminator.module.name}.pt")
                torch.save(self.optimizerD.module.state_dict(), f"{self.weight_dir}/optimizer_{self.discriminator.module.name}.pt")
            
                #generator
                torch.save(self.generator.module.state_dict(),
                           f"{self.weight_dir}/{self.generator.module.name}.pt")
                torch.save(self.optimizerD.module.state_dict(),
                           f"{self.weight_dir}/optimizer_{self.generator.module.name}.pt")
                
                if self.tracker is not None:
                    self.tracker.save(f"{self.weight_dir}/{self.discriminator.module.name}.pt")
                    self.tracker.save(f"{self.weight_dir}/{self.generator.module.name}.pt")
            else :
                #discriminaor
                torch.save(self.discriminator.state_dict(), f"{self.weight_dir}/discriminator.pt")
                torch.save(self.optimizerD.state_dict(), f"{self.weight_dir}/optimizer_discriminator.pt")
                #generator
                torch.save(self.generator.state_dict(), f"{self.weight_dir}/generator.pt")
                torch.save(self.optimizerG.state_dict(), f"{self.weight_dir}/optimizer_generator.pt")
                if self.tracker is not None:
                    self.tracker.save(f"{self.weight_dir}/{self.discriminator.name}.pt")


    def compile(self, generator,discriminator, train_loader, val_loader, optimizer: str or None, criterion: str or None,validation_steps=100):
        """
            Compile the experiment before variation

            This function simply prepare all parameters before variation the experiment

            Parameters
            ----------
            model : Subclass of torch.nn.Module
                A pytorch model
            optimizer :  String, must be a subclass of torch.optim
                -Adam
                -AdamW
                -SGD
                -etc.

            criterion : String , must be a subclass of torch.nn
                - BCEWithLogitsLoss
                - BCELoss
                - MSELoss
                - etc.

            final_activation : String, must be a subclass of torch.nn
                - Sigmoid
                - Softmax
                - etc.


        """

        # -----------model initialisation------------------------------
        self.generator = generator
        self.discriminator = discriminator
        self.validation_steps = validation_steps
        # send model to gpu
        self.generator = self.generator.to(self.device, dtype=torch.float)
        self.discriminator = self.discriminator.to(self.device, dtype=torch.float)
        print("The model has now been successfully loaded into memory")

        self.train_loader = train_loader
        self.val_loader = val_loader


        self.watch()

        self.num_classes = len(self.names)

        # if metrics is not None :
        #     self.metrics = {}
        #     for metric in metrics :
        #         assert metric in dir(skm), f"The metric {metric} is not implemented yet"
        #         self.metrics[metric] = getattr(skm, metric)
        # else :
        #     self.metrics = None
        #     logging.info("No metrics have been specified. Only the loss will be computed")

        self.metrics = Metrics(train_loader).metrics()

        if optimizer in dir(torch.optim):
            optimizer = getattr(torch.optim, optimizer)

        else:
            raise NotImplementedError("The optimizer is not implemented yet")

        signature = inspect.signature(optimizer.__init__)
        optimizer_params = {key: self.config[key] for key in list(signature.parameters.keys())[2::] if
                            key in self.config}

        self.optimizerD = optimizer(
            self.discriminator.parameters(),
            **optimizer_params
        )
        self.optimizerG = optimizer(
            self.generator.parameters(),
            **optimizer_params
        )



        if not isinstance(criterion,torch.nn.Module) :
            if criterion in dir(torch.nn):
                criterion = getattr(torch.nn, criterion)()
            elif criterion in dir(torchmetrics):
                criterion = getattr(torchmetrics, criterion)()
            else:
                raise NotImplementedError(f"The criterion {criterion} is not implemented yet")
        self.criterion = criterion

        self.schedulerD = torch.optim.lr_scheduler.OneCycleLR(self.optimizerD, max_lr=self.config["lr"],
                                                             steps_per_epoch=len(self.train_loader),
                                                             epochs=self.epoch_max)

        self.schedulerG = torch.optim.lr_scheduler.OneCycleLR(self.optimizerG, max_lr=self.config["lr"],
                                                             steps_per_epoch=len(self.train_loader),
                                                             epochs=self.epoch_max)

        
        self.metrics = Metrics(train_loader).metrics()
       

    def train(self,**kwargs):
        """
        Run the variation for the compiled experiment

        This function simply prepare all parameters before variation the experiment

        Parameters
        ----------
        **kwargs : Override default methods in compile

        """
        for key, value in kwargs.items():
            assert key in dir(self), f"You are trying to override {key}. This is not an attribute of the class Experiment"
            setattr(self,key,value)



        # if os.path.exists(f"{self.weight_dir}/{self.model.name}.pt"):
        #     print(f"Loading pretrained weights from {self.weight_dir}/{self.model.name}.pt")
        #     self.model.load_state_dict(torch.load(f"{self.weight_dir}/{self.model.name}.pt"))
        #     self.optimizer.load_state_dict(torch.load(f"{self.weight_dir}/optimizer_{self.model.name}.pt"))

        # Creates a GradScaler once at the beginning of variation.
        self.scalerD = torch.cuda.amp.GradScaler(enabled=self.config["autocast"])
        self.scalerG = torch.cuda.amp.GradScaler(enabled=self.config["autocast"])

        n, m = len(self.train_loader), len(self.val_loader)




        val_loss, results = self.validation_loop()
        self.best_loss = val_loss / m
        logging.info(f"Starting training with validation loss : {self.best_loss}")


        while self.keep_training:  # loop over the dataset multiple times

            if dist.is_initialized():
                self.train_loader.sampler.set_epoch(self.epoch)

            train_lossG,train_lossD = self.training_loop()
            if self.rank == 0:
                val_loss, results = self.validation_loop()
                self.log_metric("training_lossG", train_lossG.cpu().item() / n, epoch=self.epoch)
                self.log_metric("training_lossD", train_lossD.cpu().item() / n, epoch=self.epoch)
                self.log_metric("validation_loss", val_loss/m, epoch=self.epoch)
                self.log_metrics(results, epoch=self.epoch)





                # Finishing the loop
            self.next_epoch(val_loss/m) # the metric used to determine the best model is the validation loss here
            if self.epoch == self.epoch_max:
                self.keep_training = False
        if logging :
            logging.info("Finished Training")
        return results

    def training_loop(self):
        """

        :param model: model to train
        :param loader: training dataloader
        :param optimizer: optimizer
        :param criterion: criterion for the loss
        :param device: device to do the computations on
        :param minibatch_accumulate: number of minibatch to accumulate before applying gradient. Can be useful on smaller gpu memory
        :return: epoch loss, tensor of concatenated labels and predictions
        """
        running_lossD = 0
        running_lossG = 0
        self.generator.train()
        self.discriminator.train()
        dtype = list(self.generator.parameters())[0].dtype

        for ex,(real_images, labels) in enumerate(tqdm.tqdm(self.train_loader)):

            # ---------------- Initialization ----------------------
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()
            # send to GPU
            real_images, labels = (
                real_images.to(self.device, non_blocking=True, dtype=dtype),
                labels.to(self.device, non_blocking=True)[:,None],
            )


            # ---------------- Forward Pass ----------------------
            with torch.cuda.amp.autocast(enabled=self.autocast):

                noise = torch.randn(real_images.shape[0], 32, device=self.device) #TODO : self.config["latent_dim"] instead of hardcoded

         
                fake_images = self.generator(noise,c=labels)
                #print(real_images.shape, fake_images.shape)
                real_pred = self.discriminator(real_images,c=labels)
                fake_pred = self.discriminator(fake_images,c=labels)

            #TODO : use the criterion instead of hard coded loss
            #IF keeping WGAN ; clip norm of the weights?
            lossD = torch.mean(real_pred-fake_pred)+compute_gp(self.discriminator, real_images, fake_images,labels)
            lossG = torch.mean(fake_pred)



            # ---------------- Backward Pass ----------------------
            self.scalerG.scale(lossG).backward(retain_graph=True)
            if ex%10 == 0 :
                #TODO : only compute the real pred every ten iterations?
                self.scalerD.scale(lossD).backward()

            # ---------------- Optimizer Step ----------------------
            # Unscales the gradients of optimizer's assigned params in-place
            if self.clip_norm != 0:
                self.scalerD.unscale_(self.optimizerD)
                self.scalerG.unscale_(self.optimizerG)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.clip_norm
                )

            self.scalerD.step(self.optimizerD)
            self.scalerG.step(self.optimizerG)
            self.scalerD.update()
            self.scalerG.update()
            self.schedulerD.step()
            self.schedulerG.step()

            running_lossG += lossG.detach()
            running_lossD += lossD.detach()
            del (

                labels,
                real_images,

            )  # garbage management sometimes fails with cuda



        return running_lossG,running_lossD

    @torch.no_grad()
    def validation_loop(self):
        """

        :param model: model to evaluate
        :param loader: dataset loader
        :param criterion: criterion to evaluate the loss
        :param device: device to do the computation on
        :return: val_loss for the N epoch, tensor of concatenated labels and predictions
        """
        running_loss = 0

        self.generator.eval()

        results = {key : 0 for key in self.metrics}
        dtype = list(self.generator.parameters())[0].dtype
        for steps in range(self.validation_steps) :
            # get the inputs; data is a list of [inputs, labels]

            # send to GPU
            label = torch.randint(0, 10, (self.config["batch_size"],), device=self.device,dtype=dtype)[:,None] #TODO :remove hard coded
            noise = torch.randn(self.config["batch_size"], 32, device=self.device,dtype=dtype)


            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=self.autocast):

                images = self.generator(noise,c=label)
                fake_pred = self.discriminator(images, c=label)

            lossG = torch.mean(fake_pred)
            running_loss += lossG.detach()

            for key,metric in self.metrics.items() :
                results[key] += metric(images.float(), label.float())






            del (
                images,
                label,
                noise
            )  # garbage management sometimes fails with cuda

        return running_loss, results,



