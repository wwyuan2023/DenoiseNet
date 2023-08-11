#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decode with trained vocoder Generator."""

import os, sys
import torch, torchaudio
import yaml
import numpy as np

import denoisenet
from denoisenet.utils import load_model


class NeuralDenoiseNet(object):
    """ Neural Vocals and Accompaniment Demix """
    def __init__(self, checkpoint_path=None, config_path=None, device="cpu"):
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(denoisenet.__path__[0], "checkpoint", "checkpoint.pkl")
    
        # setup config
        if config_path is None:
            dirname = os.path.dirname(checkpoint_path)
            config_path = os.path.join(dirname, "config.yml")
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self.sampling_rate = self.config["sampling_rate"]
        
        # setup device
        self.device = torch.device(device)
        
        # setup model
        model = load_model(checkpoint_path, self.config)
        model.remove_weight_norm()
        self.model = model.eval().to(self.device)
        
        # alias inference
        self.inference = self.infer
    
    def _add_reverb(self, audio : torch.Tensor) -> torch.Tensor:
        # add reverberation
        reverberance = np.random.randint(20, 80)  # Range: (0,100)
        hf_damping = np.random.randint(20, 80)  # Range: (0,100)
        room_scale = np.random.randint(20, 80)  # Range: (0,100)
        stereo_depth = np.random.randint(20, 80)  # Range: (0,100)
        pre_delay = np.random.randint(0, 100)  # Range: (0,500)
        web_gain_db = 0  # Range: (-10,10)
        effects = [
            ["reverb", f"{reverberance}", f"{hf_damping}", f"{room_scale}",
                f"{stereo_depth}", f"{pre_delay}", f"{web_gain_db}"]
        ]
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.sampling_rate, effects)
        audio = audio[:1]        
        audio = torch.clamp(audio, -1., 1.)

        return audio
    
    @torch.no_grad()
    def infer(self, y, add_reverb=False, tta=False):
        # y: mixture of vocal and noise, dtype=Tensor, shape=(B=1,T)
        if add_reverb:
            y = self._add_reverb(y)
        x = self.model.infer(y, tta=tta)
        return x


def main():
    
    import argparse
    import logging
    import time
    import soundfile as sf
    import librosa
    
    from tqdm import tqdm
    from denoisenet.utils import find_files
    
    
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Extract vocals from noise with trained Neural DenoiseNet Generator "
                    "(See detail in denoisenet/bin/infer.py).")
    parser.add_argument("--wav-scp", "--scp", default=None, type=str,
                        help="wav.scp file. "
                             "you need to specify either wav-scp or dumpdir.")
    parser.add_argument("--dumpdir", default=None, type=str,
                        help="directory including feature files. "
                             "you need to specify either wav-scp or dumpdir.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", "--ckpt", default=None, type=str, 
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", "--conf", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--add-reverb", default=False, action='store_true',
                        help="add reverb to input wav when inference.")
    parser.add_argument("--trim-silence", "--trim-sil", default=False, action='store_true',
                        help="trim silence of header and tailer after inference.")
    parser.add_argument("--device", default="cpu", type=str,
                        help="use cpu or cuda. (default=cpu)")
    parser.add_argument("--tta", default=False, action='store_true',
                        help="use dropout when inference.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # setup model
    model = NeuralDenoiseNet(args.checkpoint, args.config, args.device)

    # setup config
    config = model.config
    
    # parse config
    sampling_rate = config["sampling_rate"]
    
    # check arguments
    if (args.wav_scp is not None and args.dumpdir is not None) or \
            (args.wav_scp is None and args.dumpdir is None):
        raise ValueError("Please specify either --dumpdir or --wav-scp.")

    # get wav files
    wav_files = dict()
    if args.dumpdir is not None:
        for filename in find_files(args.dumpdir, "*.wav"):
            utt_id = os.path.splitext(os.path.basename(filename))[0]
            wav_files[utt_id] = filename
        logging.info("From {} find {} wav files.".format(args.dumpdir, len(wav_files)))
    else:
        with open(args.wav_scp) as fid:
            for line in fid:
                line = line.strip()
                if line == "" or line[0] == "#" or line[-len(".wav"):] != ".wav": continue
                utt_id = os.path.splitext(os.path.basename(line))[0]
                wav_files[utt_id] = line
        logging.info("From {} find {} wav files.".format(args.wav_scp, len(wav_files)))
    logging.info(f"The number of wav to be denoised = {len(wav_files)}.")

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(wav_files.items(), desc="[denoise]") as pbar:
        for idx, (utt_id, wavfn) in enumerate(pbar, 1):
            start = time.time()
            
            # load pcm
            ys, sr = sf.read(wavfn, dtype=np.float32) # x: (T,C) or (T,)
            if sr != sampling_rate:
                ys = librosa.resample(ys, orig_sr=sr, target_sr=sampling_rate, axis=0)
            if ys.ndim == 1:
                ys = ys.reshape(-1, 1)
            xs = np.zeros_like(ys)
            
            for c in range(ys.shape[1]):
                # Only support mono channel
                y = ys[:, c]
                y /= abs(y).max() * 2
                
                # inference
                y = torch.from_numpy(y).view(1, -1)
                x = model.infer(y, add_reverb=args.add_reverb, tta=args.tta)
                x = x.cpu().numpy().flatten()
                x /= abs(x).max() * 2

                # trim silence
                if args.trim_silence and ys.ndim <= 1:
                    _, (xs, xe) = librosa.effects.trim(x, top_db=30)
                    xs -= int(0.05 * sampling_rate)
                    xe += int(0.05 * sampling_rate)
                    if xs < 0: xs = 0
                    x = x[xs:xe]
                
                xs[:, c] = x
            
            if xs.shape[1] == 1:
                xs = xs.flatten()
            
            # save as PCM 16 bit wav files
            sf.write(os.path.join(args.outdir, f"{utt_id}.wav"),
                xs, sampling_rate, "PCM_16")
            
            rtf = (time.time() - start) / ((xs.ndim * len(xs) * sampling_rate))
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
