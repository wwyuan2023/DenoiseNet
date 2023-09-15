#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decode with trained vocoder Generator."""

import os, sys
import torch, torchaudio
import yaml
import numpy as np
from scipy import signal

import denoisenet
from denoisenet.utils import load_model
from denoisenet import __version__


def butter_highpass_filter(x, sr, order=5, cuttoff=70):
    b, a = signal.butter(order, 2*cuttoff/sr, 'highpass')
    y = signal.filtfilt(b, a, x, axis=0)
    return y


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
        # add reverberation, audio=(c, t)
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
        audio = torch.clamp(audio, -1., 1.)

        return audio
    
    def _add_noise(self, audio : torch.Tensor, gain=0) -> torch.Tensor:
        # add gaussian white noise, audio=(c, t)
        if gain > 0:
            noise = torch.rand_like(audio) * gain
            audio += noise
            audio = torch.clamp(audio, -1., 1.)
        return audio
    
    @torch.no_grad()
    def infer(self, y, add_reverb=False, noise_scale=0, tta=False):
        # y: mixture of vocal and noise, dtype=Tensor, shape=(B, T)
        B = y.size(0)
        if add_reverb:
            y = self._add_reverb(y)[:B]
        y = self._add_noise(y, gain=noise_scale)
        y = y / y.max()
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
        description=f"Extract vocals from noise with trained Neural DenoiseNet Generator, version = {__version__} "
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
    parser.add_argument("--sampling-rate", "--sr", default=None, type=int,
                        help="target sampling rate for stored wav file.")
    parser.add_argument("--add-reverb", default=False, action='store_true',
                        help="add reverb to input wav when inference.")
    parser.add_argument("--trim-silence", "--trim-sil", default=None, type=float,
                        help="trim silence of header and tailer after inference, 45DB recommended.")
    parser.add_argument("--highpass", default=None, type=float,
                        help="highpass filter after inference.")
    parser.add_argument("--noise-scale", default=0, type=float,
                        help="gaussian white noise scale, add to input wav when inference. (default=0)")
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
            y, sr = sf.read(wavfn, dtype=np.float32) # x: (T, C) or (T,)
            if sr != sampling_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sampling_rate, axis=0)
            y = y.T if y.ndim == 2 else y.reshape(1, -1) # (B=C, T)
            y /= abs(y).max() * 2
            
            # inference
            y = torch.from_numpy(y)
            x = model.infer(y, add_reverb=args.add_reverb, noise_scale=args.noise_scale, tta=args.tta)
            x = x.cpu().numpy()
            x /= abs(x).max()

            # trim silence
            if args.trim_silence is not None:
                xs, xe = [], []
                for i in range(x.shape[0]):
                    _, (s, e) = librosa.effects.trim(x[i], top_db=args.trim_silence)
                    s -= int(0.05 * sampling_rate)
                    e += int(0.05 * sampling_rate)
                    if s < 0: s = 0
                    xs.append(s)
                    xe.append(e)
                s, e = min(xs), max(xe)
                x = x[:, s:e]

            # save as PCM 16 bit wav files
            x = x.flatten() if x.shape[0] == 1 else x.T # (T, C) or (T,)
            final_sr = sr if args.sampling_rate is None else args.sampling_rate
            if final_sr != sampling_rate:
                x = librosa.resample(x, orig_sr=sampling_rate, target_sr=final_sr, res_type="scipy", axis=0)
            
            if args.highpass is not None:
                x = butter_highpass_filter(x, final_sr, cuttoff=args.highpass)
            
            sf.write(os.path.join(args.outdir, f"{utt_id}.wav"),
                x, final_sr, "PCM_16")
            
            rtf = (time.time() - start) / (len(x) / final_sr)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
