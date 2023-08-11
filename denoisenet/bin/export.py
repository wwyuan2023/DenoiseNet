#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Parallel WaveGAN."""

import argparse
import logging
import os
import sys
import glob

import numpy as np
import torch
import yaml

import denoisenet
import denoisenet.models
from denoisenet.utils import load_model


def find_checkpoint_paths(dir_path, regex="checkpoint*.pkl"):
    path_list = glob.glob(os.path.join(dir_path, regex))
    path_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return path_list

def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Export DenoiseNet (See detail in denoisenet/bin/export.py).")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--checkpoint", "--ckpt", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", "--conf", default=None, type=str,
                        help="yaml format configuration file.")
    parser.add_argument("--greedy-soup", "--greedy", default=False, action='store_true',
                        help="use average of checkpoints.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
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
    
    # load config
    config_path = args.config
    if config_path is None:
        dirname = args.checkpoint if os.path.isdir(args.checkpoint) else \
            os.path.dirname(args.checkpoint)
        config_path = os.path.join(dirname, "config.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # load model
    if not os.path.isdir(args.checkpoint):
        model = load_model(args.checkpoint, config)
    else:
        ckpt_paths = find_checkpoint_paths(args.checkpoint)
        model = load_model(ckpt_paths[-1], config)
        if args.greedy_soup and len(ckpt_paths) > 1:
            avg = model.state_dict()
            for ckpt in ckpt_paths[:-1]:
                logging.info(f"Load [{ckpt}] for averaging.")
                states = load_model(ckpt, config).state_dict()
                for k in avg.keys():
                    avg[k] += states[k]
            # average
            for k in avg.keys():
                avg[k] = torch.true_divide(avg[k], len(ckpt_paths))
            model.load_state_dict(avg)
    
    # save config to outdir
    config["version"] = denoisenet.__version__   # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    
    # print parameters
    logging.info(model)
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total parameters: {total_params}")
    
    # save model to outdir
    checkpoint_path = os.path.join(args.outdir, "checkpoint.pkl")
    state_dict = {
        "model": model.state_dict()
    }
    torch.save(state_dict, checkpoint_path)
    logging.info(f"Successfully export model parameters from [{args.checkpoint}] to [{checkpoint_path}].")


if __name__ == "__main__":
    main()
