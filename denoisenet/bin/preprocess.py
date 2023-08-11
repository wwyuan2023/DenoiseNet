# -*- coding: utf-8 -*-

"""Perform preprocessing and raw feature extraction."""

import os
import argparse
import logging

import librosa
import soundfile as sf

from scipy import signal
from tqdm import tqdm

from denoisenet.utils import find_files


def butter_highpass_filter(x, sr, order=5, cuttoff=70):
    b, a = signal.butter(order, 2*cuttoff/sr, 'highpass')
    y = signal.filtfilt(b, a, x)
    return y


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in denoisenet/bin/preprocess.py).")
    parser.add_argument("--wav-scp", "--scp", default=None, type=str,
                        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.")
    parser.add_argument("--rootdir", default=None, type=str,
                        help="directory including wav files. you need to specify either scp or rootdir.")
    parser.add_argument("--dumpdir", type=str, required=True,
                        help="directory to dump wav files.")
    parser.add_argument("--sampling_rate", type=int, default=16000,
                        help="sampling rate for dump wav files. (default=16000)")
    parser.add_argument("--highpass", type=float, default=70.0,
                        help="highpass frequency using `signal.butter`. (default=0)")
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
        logging.warning('Skip DEBUG/INFO messages')
    
    # set useful vars
    sampling_rate = args.sampling_rate
    highpass = args.highpass

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or \
            (args.wav_scp is None and args.rootdir is None):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get wav files
    wav_files = dict()
    if args.rootdir is not None:
        for filename in find_files(args.rootdir, "*.wav"):
            utt_id = filename.replace("/", "_")
            wav_files[utt_id] = filename
        logging.info("From {} find {} wav files.".format(args.rootdir, len(wav_files)))
    else:
        with open(args.wav_scp) as fid:
            for line in fid:
                line = line.strip()
                if line == "" or line[0] == "#" or line[-len(".wav"):] != ".wav": continue
                utt_id = line.replace("/", "_")
                wav_files[utt_id] = line
        logging.info("From {} find {} wav files.".format(args.wav_scp, len(wav_files)))

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)
    
    # process each data
    for utt_id, wavfn in tqdm(wav_files.items()):
        audio, fs = sf.read(wavfn) # audio: (T,C) or (T,)
        
        # check sampling rate
        if fs < sampling_rate:
            logging.info("The sampling rate of [{}] is {} < {}".format(wavfn, fs, sampling_rate))
            continue
        elif fs > sampling_rate:
            audio = librosa.resample(audio.T, orig_sr=fs, target_sr=sampling_rate).T
        
        # normalization
        audio = audio / max(abs(audio)) * 0.5

        # highpass filter
        if highpass > 0:
            audio = butter_highpass_filter(audio, sampling_rate, cuttoff=highpass)
                
        # save wav
        wavfn = os.path.join(args.dumpdir, f"{utt_id}")
        sf.write(wavfn, audio, sampling_rate, "PCM_16")
        
    print("Done!")


if __name__ == "__main__":
    main()

