# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import torch, torchaudio

from torch.utils.data import Dataset


def _load_scpfn(filename):
    scplist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                scplist.append(line)
    return scplist


class AudioSCPDataset(Dataset):
    """PyTorch compatible audio dataset based on scp files."""

    def __init__(
        self,
        vocal_scpfn,
        noise_scpfn,
        segment_size,
        sampling_rate_needed=16000,
        hop_size=160,
        reverb_rate=0.1,
        vocal_mixup_rate=0.1,
        noise_mixup_rate=0.1,
        return_utt_id=False,
        num_repeat=100,
    ):
        """Initialize dataset.
        """
        assert segment_size > 0
        segment_size = (segment_size // hop_size) * hop_size
        
        self.sampling_rate_needed = int(sampling_rate_needed)
        self.segment_size_needed = int(segment_size)
        self.segment_size = int(segment_size + (sampling_rate_needed//2)*2) # More than 0.5 second at both head and tail of segment
        def _filter_audiofile(filename):
            if filename == "": return False
            meta = torchaudio.info(filename)
            if meta.sample_rate != self.sampling_rate_needed:
                return False
            return True

        def _repeat_list(_list):
            relist = []
            for _ in range(num_repeat):
                for p in _list:
                    relist.append(p)
            return relist
        
        vocal_list = []
        ndrop = 0
        for wavfn in _load_scpfn(vocal_scpfn):
            if _filter_audiofile(wavfn):
                vocal_list.append(wavfn)
                continue
            logging.warning(f"Drop files, vocal file={wavfn}")
            ndrop += 1
        logging.warning(f"Vocal: {ndrop} files are dropped, and ({len(vocal_list)}) files are loaded.")

        noise_list = []
        ndrop = 0
        for wavfn in _load_scpfn(noise_scpfn):
            if _filter_audiofile(wavfn):
                noise_list.append(wavfn)
                continue
            logging.warning(f"Drop files, noise file={wavfn}")
            ndrop += 1
        logging.warning(f"Noise: {ndrop} files are dropped, and ({len(noise_list)}) files are loaded.")
        
        self.vocal_list = _repeat_list(vocal_list)
        self.noise_list = _repeat_list(noise_list)
        self.num_vocal = len(self.vocal_list)
        self.num_noise = len(self.noise_list)
        self.return_utt_id = return_utt_id

        self.volmax = dict() # Record max sample of each audio, including vocal and noise.
        self.durmax = dict() # Record duration of each audio, same as `self.volmax`.
        
        self.hop_size = hop_size
        self.reverb_rate = reverb_rate
        self.vocal_mixup_rate = vocal_mixup_rate
        self.noise_mixup_rate = noise_mixup_rate
        self.cache_vocal = [None for _ in range(1000)] # Caching segment for mixing up different source
        self.cache_noise = [None for _ in range(1000)] # Same as above
    
    def _load_audio(self, filename):
        # Load pcm
        if self.durmax.get(filename, -1) > self.segment_size:
            offset = np.random.randint(0, self.durmax[filename] - self.segment_size)
            x, sr = torchaudio.load(filename, frame_offset=offset, num_frames=self.segment_size)
        else:
            x, sr = torchaudio.load(filename)  # load all samples
            self.volmax[filename] = max(x.abs().max().item(), 1e-5) # safe guard
            self.durmax[filename] = x.size(1)
        
        # Check `sr` and pad `x`: (c,t)
        assert sr == self.sampling_rate_needed
        while x.size(1) < self.segment_size:
            x = torch.cat((x,x), dim=1)
        
         # Randomly select one channel and Maximize volume
        c = np.random.randint(0, x.size(0)) 
        if x.size(1) != self.segment_size:
            offset = np.random.randint(0, x.size(1) - self.segment_size)
            x = x[c:c+1, offset:offset+self.segment_size] / self.volmax[filename]
        else:
            x = x[c:c+1] / self.volmax[filename]

        return x  # x: (1,t)
    
    def _vocal_data_augment(self, vocal):
        # vocal (Tensor): (1,T), where more than 0.5 sec at head and tail, its will be cutted off before callback return
        clean = vocal.clone()
        
        # Add reverberation
        if np.random.uniform() < self.reverb_rate:
            reverberance = np.random.randint(20, 80) # Range: (0,100)
            hf_damping = np.random.randint(20, 80) # Range: (0,100)
            room_scale = np.random.randint(20, 80) # Range: (0,100)
            stereo_depth = np.random.randint(20, 80) # Range: (0,100)
            pre_delay = np.random.randint(0, 100) # Range: (0,500)
            web_gain_db = 0 # Range: (-10,10)
            effects = [
                ["reverb", f"{reverberance}", f"{hf_damping}", f"{room_scale}", f"{stereo_depth}", f"{pre_delay}", f"{web_gain_db}"]
            ]
            vocal, _ = torchaudio.sox_effects.apply_effects_tensor(vocal, self.sampling_rate_needed, effects)
        
        # Homologous mixture
        elif self.vocal_mixup_rate > 0:
            idx = np.random.randint(0, len(self.cache_vocal))
            if self.cache_vocal[idx] is None:
                self.cache_vocal[idx] = vocal.clone()
            elif np.random.uniform() < self.vocal_mixup_rate:
                _, _, vocal, alpha = self._do_mix(vocal, self.cache_vocal[idx], alpha=np.random.uniform(0.9, 1.0))
                clean *= alpha[0]
            elif np.random.uniform() < 0.5:
                _, _, self.cache_vocal[idx], _ = self._do_mix(vocal, self.cache_vocal[idx], alpha=np.random.uniform(0.0, 0.5))
            else:
                self.cache_vocal[idx] = None
                
        return vocal, clean
    
    def _noise_data_augment(self, noise):
        # noise (Tensor): (1,T)

        # Homologous mixture
        if self.noise_mixup_rate > 0:
            idx = np.random.randint(0, len(self.cache_noise))
            if self.cache_noise[idx] is None:
                self.cache_noise[idx] = noise.clone()
            elif np.random.uniform() < self.noise_mixup_rate:
                _, _, noise, _ = self._do_mix(noise, self.cache_noise[idx], alpha=np.random.uniform(0.7, 1.0))
            elif np.random.uniform() < 0.5:
                _, _, self.cache_noise[idx], _ = self._do_mix(noise, self.cache_noise[idx], alpha=np.random.uniform(0.0, 0.5))
            else:
                self.cache_noise[idx] = None
        
        return noise
    
    def _do_mix(self, src1, src2, alpha=0.5):
        a1, a2 = np.sqrt(alpha), np.sqrt(1 - alpha)
        src1 *= a1
        src2 *= a2
        mix = torch.clamp(src1 + src2, min=-1., max=1.)
        return src1, src2, mix, (a1, a2)
    
    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).

        """
        vocalfn = self.vocal_list[idx]
        noisefn = self.noise_list[np.random.randint(0, self.num_noise)]

        # Read pcm
        vocal = self._load_audio(vocalfn)
        noise = self._load_audio(noisefn)

        # Data augmentation
        vocal, clean = self._vocal_data_augment(vocal)
        noise = self._noise_data_augment(noise)

        # More than 0.5 sec at head and tail, its should be cutted off, and also delete channel dim
        clean = clean[0, self.sampling_rate_needed//2:-self.sampling_rate_needed//2]
        vocal = vocal[0, self.sampling_rate_needed//2:-self.sampling_rate_needed//2]
        noise = noise[0, self.sampling_rate_needed//2:-self.sampling_rate_needed//2]

        # Mix vocal and noise, shape=(T,)
        if np.random.uniform() < 0.75:
            _, _, mixture, alpha = self._do_mix(vocal, noise, alpha=np.random.uniform(0.6, 1.0))
            clean *= alpha[0]
        else:
            mixture = vocal
        
        assert clean.size(0) == mixture.size(0) == self.segment_size_needed

        # clip value
        clean = torch.clamp(clean, -1., 1.)
        mixture = torch.clamp(mixture, -1., 1.)

        if self.return_utt_id:
            utt_id = os.path.splitext(os.path.basename(vocalfn))[0]
            items = utt_id, clean, mixture
        else:
            items = clean, mixture

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return self.num_vocal


