# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BandSplit(nn.Module):
    def __init__(self, band_widths, N=128):
        super().__init__()
        self.N = N
        self.band_widths = band_widths
        self.K = len(band_widths)
        self.norm_layers = nn.ModuleList([nn.LayerNorm((self.band_widths[i] * 2)) for i in range(self.K)])
        self.fc_layers = nn.ModuleList([nn.Linear(self.band_widths[i] * 2, self.N) for i in range(self.K)])

    def forward(self, x):
        # x: (B, F, T, 2)
        B, F, T, _ = x.size()
        x = x.permute(0, 2, 1, 3).view(B, T, F*2) # (B, T, F*2)
        outs = []
        start, end = 0, self.band_widths[0] * 2
        for i in range(self.K):
            end = start + self.band_widths[i] * 2
            suband = x[..., start:end]
            suband = self.norm_layers[i](suband)
            suband = self.fc_layers[i](suband) # (B, T, N)
            outs.append(suband)
            start = end
        assert F*2 == end, f"F={F}, end={end}"
        z = torch.stack(outs, dim=2) # (B, T, K, N)
        return z


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim*2),
            nn.GLU()
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x


class _MaskEstimation(nn.Module):
    def __init__(self, band_widths, N):
        super().__init__()
        self.K = len(band_widths)
        self.band_widths = band_widths
        self.norm_layers = nn.ModuleList([nn.LayerNorm(N) for _ in self.band_widths])
        self.mlp_layers = nn.ModuleList([_MLP(N, b*2, N*4) for b in self.band_widths])

    def forward(self, Q):
        # Q: (B, N, K, T)
        B, N, K, T = Q.size()
        Q = Q.permute(2, 0, 3, 1) # (K, B, T, N)
        outs = []
        for i in range(self.K):
            suband = self.norm_layers[i](Q[i])
            suband = self.mlp_layers[i](suband) # (B, T, b*2)
            outs.append(suband.view(B, T, self.band_widths[i], 2)) # (B, T, b, 2)
        mask = torch.cat(outs, dim=2).transpose(1, 2) # (B, F, T, 2)
        return mask


class _BandSplitRNNBlock(nn.Module):
    def __init__(self, N=128, hidden_size=256, groups=8, dropout=0.1):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, N)

        self.seq_blstm = nn.LSTM(N, hidden_size, batch_first=True, dropout=dropout, bidirectional=True)
        self.seq_linear = nn.Linear(hidden_size*2, N)

        self.band_blstm = nn.LSTM(N, hidden_size, batch_first=True, dropout=dropout, bidirectional=True)
        self.band_linear = nn.Linear(hidden_size*2, N)

    def forward(self, Z):
        # Z : (B, N, K, T)
        B, N, K, T = Z.size()

        # ------first we do the sequence level module------
        X = Z.permute(0, 2, 1, 3).reshape(B*K, N, T)
        X = self.group_norm(X)
        X = X.transpose(1, 2) # (B*K, T, N)
        X, _ = self.seq_blstm(X) # (B*K, T, hidden*2)
        X = self.seq_linear(X) # (B*K, T, N)
        X = X.view(B, K, T, N).permute(0, 3, 1, 2).contiguous() # (B, N, K, T)
        Z = Z + X

        # ------second we do the band level module------
        X = Z.permute(0, 3, 1, 2).reshape(B*T, N, K)
        X = self.group_norm(X)
        X = X.transpose(1, 2) # (B*T, K, N)
        X, _ = self.band_blstm(X) # (B*T, K, hidden*2)
        X = self.band_linear(X) # (B*T, K, N)
        X = X.view(B, T, K, N).permute(0, 3, 2, 1).contiguous() # (B, N, K, T)
        Z = Z + X

        return Z


class _BandSplitRNN(nn.Module):
    def __init__(self, N, K, num_layers=12, hidden_size=256, groups=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.rnns = nn.ModuleList([_BandSplitRNNBlock(N, hidden_size, groups, dropout) for i in range(num_layers)])
        
    def forward(self, Z):
        for i in range(self.num_layers):
            Z = self.rnns[i](Z)
        return Z


class BSRNN(nn.Module):
    def __init__(
        self,
        sampling_rate=16000,
        fft_size=1024,
        hop_size=256,
        band_widths=(5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32),
        num_channels=128,
        num_layers=12,
        use_weight_norm=False,
    ):
        super().__init__()
        assert sum(band_widths) == fft_size // 2 + 1
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.band_widths = band_widths
        self.N = num_channels
        self.K = len(band_widths)
        self.num_layers = num_layers
        
        self.register_buffer('window', torch.hann_window(fft_size), persistent=False)
        
        # band split
        self.split = _BandSplit(self.band_widths[:], num_channels)
        
        # encoder
        self.rnn = _BandSplitRNN(self.N, self.K, self.num_layers, self.N*2)
        
        # output
        self.masks = _MaskEstimation(self.band_widths, num_channels)
                
        # reset parameters
        self.reset_parameters()
        
        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, 0.0, 0.2)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        self.apply(_reset_parameters)
        
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
    
    def _stft(self, x):
        # (B, t) -> (B, F, T, 2), F=fft_size//2+1, T=t//hop_size+1
        return torch.stft(x,
            n_fft=self.fft_size, hop_length=self.hop_size, 
            win_length=self.fft_size, window=self.window,
            center=True, pad_mode='reflect', return_complex=False)
    
    def _istft(self, s):
        # (B, F, T, 2) -> (B, t), F=fft_size//2+1, t=(T-1)*hop_size
        return torch.istft(torch.complex(s[..., 0].contiguous(), s[..., 1].contiguous()),
            n_fft=self.fft_size, hop_length=self.hop_size, 
            win_length=self.fft_size, window=self.window,
            center=True, return_complex=False)
    
    def forward(self, x, y):
        # x: source signal, shape=(B, t), t=audio length
        # y: mixture signal with same shape of `x`
        
        # stft
        sx = self._stft(x) # (B, F, T, 2)
        sy = self._stft(y) # (B, F, T, 2)
        
        # magnitude of source `x`
        mx = torch.norm(sx, p=2, dim=-1) # (B, F, T)
        
        # band split
        X = self.split(sy) # (B, T, K, N)
        
        # rnn
        Q = self.rnn(X.transpose(1, 3)) # (B, N, K, T)
        
        # mask estimation
        M = self.masks(Q) # (B, F, T, 2)
        
        # apply mask
        sx_hat = sy * M
        mx_hat = torch.norm(sx_hat, p=2, dim=-1) # (B, F, T)
        
        # istft
        x_hat = self._istft(sx_hat) # (B, t)
        
        return x_hat, (mx, mx_hat), (sx, sx_hat)
    
    @torch.no_grad()
    def infer(self, y, tta=False, segment_seconds=4):
        # y: mixture signal, shape=(B, t), t=audio length
        assert segment_seconds > 2       
        segment_length = int(segment_seconds * self.sampling_rate)
        hop_length = int(1 * self.sampling_rate)
        overlap_length = segment_length - hop_length
        window = torch.hann_window(overlap_length*2).to(y.device)
        inc_window, dec_window = window[:overlap_length].unsqueeze(0), window[overlap_length:].unsqueeze(0)
        
        # valid length
        batch_size, orig_length = y.size()
        pad_length = overlap_length - (orig_length % segment_length)
        if pad_length < 0: pad_length += segment_length
        pad0 = pad_length // 2
        pad1 = pad_length - pad0
        y = F.pad(y, (pad0, pad1))
        
        # split segments
        start, end = 0, segment_length
        segments = []
        while end <= y.size(1):
            segment = y[:, start:end]
            start = start + hop_length
            end = end + hop_length
            segments.append(segment)
        assert end - hop_length == y.size(1), f"{end}, {y.size()}"
        ys = torch.stack(segments, dim=0) # (S, B, L)
        ys = ys.view(-1, ys.size(-1)) # (S*B, L)
        
        # stft
        sy = self._stft(ys)
        
        # band split
        X = self.split(sy)
        
        # rnn
        Q = self.rnn(X.transpose(1, 3))
        
        # mask estimation
        M = self.masks(Q)
        
        # apply mask
        sx_hat = sy * M
        
        # istft
        xs_hat = self._istft(sx_hat) # (S*B, L)
        
        # combine segments
        assert xs_hat.size(1) == segment_length, f"{segment_length}, {xs_hat.size()}"
        xs_hat = xs_hat.view(-1, batch_size, segment_length) # (S, B, L)
        output = torch.zeros_like(y)
        output[:, :segment_length] = xs_hat[0]
        start = hop_length
        for i in range(1, xs_hat.size(0)):
            output[:, start:start+overlap_length] *= dec_window
            xs_hat[i, :, :overlap_length] *= inc_window
            output[:, start:start+segment_length] += xs_hat[i]
            start += hop_length
        
        pad1 = None if pad1 == 0 else -pad1
        x_hat = output[:, pad0:pad1]
        assert x_hat.size(-1) == orig_length, f"{orig_length}, {x_hat.size()}"
        
        return x_hat


