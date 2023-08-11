#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Model."""

import os
import sys
import argparse
import logging

import matplotlib
import numpy as np
import soundfile as sf
import yaml
import torch
import torch.nn as nn

from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import denoisenet
import denoisenet.models
import denoisenet.optimizers
import denoisenet.lr_scheduler

from denoisenet.datasets import AudioSCPDataset
from denoisenet.layers import MultiResolutionSTFTLoss
from denoisenet.utils import eval_sdr

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" models.
            criterion (dict): Dict of criterions. It must contrain "ce" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" optimizer.
            scheduler (dict): Dict of schedulers. It must contrain "generator" scheduler.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        
    def run(self):
        """Run training."""
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])
    
    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x = batch[0].to(self.device) # clean vocal, (B, T)
        y = batch[1].to(self.device) # mixture, (B, T)
    
        x_, (m, m_), (s, s_) = self.model(x, y)
    
        # PCM mae loss
        pcm_loss = self.criterion["mae"](x_, x[..., :x_.size(-1)])
        self.total_train_loss["train/pcm_loss"] += pcm_loss.item()
        gen_loss = self.config["lambda_pcm"] * pcm_loss
                
        # magnitude mae loss
        mag_loss = self.criterion["mae"](m_, m)
        self.total_train_loss["train/mag_loss"] += mag_loss.item()
        gen_loss += self.config["lambda_mag"] * mag_loss
        
        # spectrum loss
        if self.config.get("lambda_spec", 0) > 0:
            spec_loss = self.criterion["mae"](s_, s)
            self.total_train_loss["train/spec_loss"] += spec_loss.item()
            gen_loss += self.config["lambda_spec"] * spec_loss
        
        # record sdr
        sdr = eval_sdr(x_.detach().unsqueeze(1), x[..., :x_.size(-1)].unsqueeze(1))
        self.total_train_loss["train/sdr"] += sdr.item()
        
        # total loss
        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["generator_grad_norm"]
            )
        self.optimizer.step()
        self.scheduler.step()
        
        # update counts
        self.steps += 1
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1

        lr_per_epoch = self.scheduler.get_last_lr()[0]
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training ({train_steps_per_epoch} steps per epoch), "
                     f"and learning rate = {lr_per_epoch:.6f}")

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x = batch[0].to(self.device)
        y = batch[1].to(self.device)
        
        x_, (m, m_), (s, s_) = self.model(x, y)
    
        # PCM mae loss
        pcm_loss = self.criterion["mae"](x_, x[..., :x_.size(-1)])
        self.total_eval_loss["eval/pcm_loss"] += pcm_loss.item()
        gen_loss = self.config["lambda_pcm"] * pcm_loss
                
        # magnitude mae loss
        mag_loss = self.criterion["mae"](m_, m)
        self.total_eval_loss["eval/mag_loss"] += mag_loss.item()
        gen_loss += self.config["lambda_mag"] * mag_loss
        
        # spectrum loss
        if self.config.get("lambda_spec", 0) > 0:
            spec_loss = self.criterion["mae"](s_, s)
            self.total_eval_loss["eval/spec_loss"] += spec_loss.item()
            gen_loss += self.config["lambda_spec"] * spec_loss
        
        # record sdr
        sdr = eval_sdr(x_.unsqueeze(1), x[..., :x_.size(-1)].unsqueeze(1))
        self.total_eval_loss["eval/sdr"] += sdr.item()
        
        # total loss
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
    
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        # calculate loss for each batch
        num_save_results = 0
        for eval_steps_per_epoch, batch in enumerate(self.data_loader["dev"], 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if num_save_results < self.config["num_save_intermediate_results"]:
                num_save_results = self._genearete_and_save_intermediate_result(batch, num_save_results)

        logging.info(f"(Steps: {self.steps}) Finished evaluation ({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")
        
        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        self.model.train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch, num_save_results):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt
        
        # generate
        x = batch[0].to(self.device)
        y = batch[1].to(self.device)
        x_, _, _ = self.model(x, y)
        x = x[..., :x_.size(-1)]
        y = y[..., :x_.size(-1)]
        
        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (g, r, m) in enumerate(zip(x_, x, y), 1):
            # convert to ndarray
            g = g.view(-1).cpu().numpy().flatten() # generated
            r = r.view(-1).cpu().numpy().flatten() # groundtruth
            m = m.view(-1).cpu().numpy().flatten() # mixture
            
            # clip to [-1, 1]
            g = np.clip(g, -1, 1)
            r = np.clip(r, -1, 1)
            m = np.clip(m, -1, 1)
            
            # plot figure and save it
            num_save_results += 1
            figname = os.path.join(dirname, f"{num_save_results}.png")
            plt.subplot(3, 1, 1)
            plt.plot(m)
            plt.title("mixture")
            plt.subplot(3, 1, 2)
            plt.plot(r)
            plt.title("groundtruth speech")
            plt.subplot(3, 1, 3)
            plt.plot(g)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
            
            # save as wavfile
            sr = self.config["sampling_rate"]
            sf.write(figname.replace(".png", "_mix.wav"), m, sr, "PCM_16")
            sf.write(figname.replace(".png", "_ref.wav"), r, sr, "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), g, sr, "PCM_16")
            
            if num_save_results >= self.config["num_save_intermediate_results"]:
                break
        
        return num_save_results

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train DenoiseNet (See detail in denoisenet/bin/train.py).")
    parser.add_argument("--train-scp", type=str, required=True,
                        help="train.scp file for training. .")
    parser.add_argument("--dev-scp", type=str, required=True,
                        help="valid.scp file for validation. ")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = denoisenet.__version__   # add version info
    if args.rank == 0:
        with open(os.path.join(args.outdir, "config.yml"), "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)
        for key, value in config.items():
            logging.info(f"{key} = {value}")
    
    # valid length
    if config.get("downsample_factors", None) is not None:
        assert config["batch_max_steps"] % (np.prod(config["downsample_factors"]) * config["hop_size"]) == 0

    # get dataset
    dataset_params = {
        "segment_size": config["batch_max_steps"],
        "sampling_rate_needed": config["sampling_rate"],
        "hop_size": config["hop_size"],
        "reverb_rate": config.get("reverb_rate", 0),
        "vocal_mixup_rate": config.get("vocal_mixup_rate", 0),
        "noise_mixup_rate": config.get("noise_mixup_rate", 0),
    }
    vocal_scpfn, noise_scpfn = args.train_scp.strip(',').split(',')
    train_dataset = AudioSCPDataset(vocal_scpfn, noise_scpfn, **dataset_params)
    logging.info(f"The number of training files = {len(train_dataset)}.")
    vocal_scpfn, noise_scpfn = args.dev_scp.strip(',').split(',')
    dev_dataset = AudioSCPDataset(vocal_scpfn, noise_scpfn, **dataset_params)
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            batch_size=config["batch_size"],
            num_workers=1,
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    generator_class = getattr(denoisenet.models, config["generator_type"])
    model = generator_class(**config["generator_params"]).to(device)
    logging.info(model)
    
    # print parameters
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for param in model.parameters():
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Non-trainable parameters: {nontrainable_params}\n")
    
    # define criterion and optimizers
    criterion = {
        "stft": MultiResolutionSTFTLoss(**config["stft_loss_params"]).to(device),
        "mae": nn.L1Loss().to(device),
        "mse": nn.MSELoss().to(device),
    }
    
    generator_optimizer_class = getattr(denoisenet.optimizers, config["generator_optimizer_type"])
    optimizer = generator_optimizer_class(
        model.parameters(),
        **config["generator_optimizer_params"],
    )

    generator_scheduler_class = getattr(denoisenet.lr_scheduler, config["generator_scheduler_type"])
    scheduler = generator_scheduler_class(
        optimizer=optimizer,
        **config["generator_scheduler_params"],
    )
    
    if args.distributed:
        # wrap model for distributed training
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")
    
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        logging.info(f"KeyboardInterrupt @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
