# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# refer to: https://arxiv.org/pdf/2109.05418.pdf

class _RCBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=(3,3), act_func="LeakyReLU", act_params={}):
        super().__init__()
        assert len(kernel_size) > 1 and all([ k % 2 == 1 for k in kernel_size ])
        padding = [ k // 2 for k in kernel_size ]
        inner_channels = max(16, in_channels//2)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            getattr(torch.nn, act_func)(**act_params),
            nn.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(inner_channels),
            getattr(torch.nn, act_func)(**act_params),
            nn.Conv2d(inner_channels, in_channels, kernel_size=kernel_size, padding=padding),
        )
        self.skiper = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))
        
    def forward(self, x):
        # x: (B, C, T, F)
        return self.skiper(x) + self.layers(x) # (B, C, T, F)


class _Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2, act_func="LeakyReLU", act_params={}):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            getattr(torch.nn, act_func)(**act_params),
            nn.Conv2d(in_channels, out_channels, factor, stride=factor),
        )
        
    def forward(self, x):
        # x: (B, Cin, T, F)
        return self.conv(x) # (B, Cout, T//factor, F//factor)


class _Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2, act_func="LeakyReLU", act_params={}):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            getattr(torch.nn, act_func)(**act_params),
            nn.ConvTranspose2d(in_channels, out_channels, factor, stride=factor),
        )
        
    def forward(self, x):
        # x: (B, Cin, T, F)
        return self.conv(x) # (B, Cout, T*factor, F*factor)


class _REBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), factor=2, act_func="LeakyReLU", act_params={}):
        super().__init__()
        act_kwargs = {"act_func": act_func, "act_params": act_params}
        self.blocks = nn.Sequential(
            _RCBlock(in_channels, kernel_size, **act_kwargs),
            _RCBlock(in_channels, kernel_size, **act_kwargs),
            _RCBlock(in_channels, kernel_size, **act_kwargs),
            _RCBlock(in_channels, kernel_size, **act_kwargs),
        )
        self.downsample = _Downsample(in_channels, out_channels, factor=factor, **act_kwargs)
    
    def forward(self, x):
        # x: (B, Cin, T, F)
        e = self.blocks(x) # (B, Cin, T, F)
        x = self.downsample(e) # (B, Cout, T//factor, F//factor)
        return x, e


class _RDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), factor=2, act_func="LeakyReLU", act_params={}):
        super().__init__()
        act_kwargs = {"act_func": act_func, "act_params": act_params}
        self.upsample = _Upsample(in_channels, out_channels, factor=factor, **act_kwargs)
        self.blocks = nn.Sequential(
            _RCBlock(out_channels, kernel_size, **act_kwargs),
            _RCBlock(out_channels, kernel_size, **act_kwargs),
            _RCBlock(out_channels, kernel_size, **act_kwargs),
            _RCBlock(out_channels, kernel_size, **act_kwargs),
        )
    
    def forward(self, x, e):
        # x: (B, Cin, T, F)
        # e: (B, Cout, T*factor, F*factor)
        x = self.upsample(x) # (B, Cout, T*factor, F*factor)
        x = self.blocks(x + e) # (B, Cout, T*factor, F*factor)
        return x


class _ICBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=(3,3), act_func="LeakyReLU", act_params={}):
        super().__init__()
        act_kwargs = {"act_func": act_func, "act_params": act_params}
        self.blocks = nn.Sequential(
            _RCBlock(in_channels, kernel_size, **act_kwargs),
            _RCBlock(in_channels, kernel_size, **act_kwargs),
            _RCBlock(in_channels, kernel_size, **act_kwargs),
            _RCBlock(in_channels, kernel_size, **act_kwargs),
        )
    
    def forward(self, x):
        return self.blocks(x) # (B, C, T, F)


class ResUNetDecouple(nn.Module):
    def __init__(
        self,
        sampling_rate=16000,
        fft_size=1024,
        hop_size=256,
        downsample_factors=(2, 2, 2, 2, 2, 2),
        use_weight_norm=False,
    ):
        super().__init__()
        assert len(downsample_factors) == 6
        assert (fft_size//2) % np.prod(downsample_factors) == 0
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.downsample_factors = downsample_factors
        
        self.register_buffer('window', torch.hann_window(fft_size), persistent=False)
        
        # prenet
        self.prenet = nn.Sequential(
            nn.Conv2d( 2, 32, 1),
            nn.LayerNorm(fft_size//2),
            nn.Conv2d(32, 32, 1),
            nn.LayerNorm(fft_size//2),
        )
        
        # encoder
        self.encoder1 = _REBlock( 32,  64, factor=downsample_factors[0], act_func="LeakyReLU")
        self.encoder2 = _REBlock( 64, 128, factor=downsample_factors[1], act_func="LeakyReLU")
        self.encoder3 = _REBlock(128, 192, factor=downsample_factors[2], act_func="LeakyReLU")
        self.encoder4 = _REBlock(192, 256, factor=downsample_factors[3], act_func="LeakyReLU")
        self.encoder5 = _REBlock(256, 384, factor=downsample_factors[4], act_func="LeakyReLU")
        self.encoder6 = _REBlock(384, 512, factor=downsample_factors[5], act_func="LeakyReLU")
        
        # bottleneck
        self.bottle1 = _ICBlock(512, act_func="ReLU")
        self.bottle2 = _ICBlock(512, act_func="ReLU")
        self.bottle3 = _ICBlock(512, act_func="ReLU")
        self.bottle4 = _ICBlock(512, act_func="ReLU")
        
        # decoder
        self.decoder6 = _RDBlock(512, 384, factor=downsample_factors[5], act_func="LeakyReLU")
        self.decoder5 = _RDBlock(384, 256, factor=downsample_factors[4], act_func="LeakyReLU")
        self.decoder4 = _RDBlock(256, 192, factor=downsample_factors[3], act_func="LeakyReLU")
        self.decoder3 = _RDBlock(192, 128, factor=downsample_factors[2], act_func="LeakyReLU")
        self.decoder2 = _RDBlock(128,  64, factor=downsample_factors[1], act_func="LeakyReLU")
        self.decoder1 = _RDBlock( 64,  32, factor=downsample_factors[0], act_func="LeakyReLU")
        
        # postnet
        self.postnet = nn.ModuleList()
        for kernel_size in (3, 5, 7):
            padding = int(kernel_size / 2)
            self.postnet.append(
                nn.Sequential(
                    nn.LayerNorm(fft_size//2),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 32, kernel_size, padding=padding),
                    nn.LayerNorm(fft_size//2),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 32, kernel_size, padding=padding),
                )
            )
        
        # output
        self.mask = nn.Conv2d(32, 1, 1)
        self.twiddle = nn.Conv2d(32, 2, 1)
        
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
        sx = self._stft(x)[:, :-1, :-1].contiguous() # (B, F-1, T-1, 2)
        sy = self._stft(y)[:, :-1, :-1].contiguous() # (B, F-1, T-1, 2)
        
        # magnitude and phase
        mx = torch.norm(sx, p=2, dim=-1) # (B, F-1, T-1)
        my = torch.norm(sy, p=2, dim=-1) # (B, F-1, T-1)
        py = F.normalize(sy, p=2, dim=-1, eps=1e-7) # (B, F-1, T-1, 2)
        
        # prenet
        x0 = self.prenet(sy.transpose(1,3)) # <- (B, 2, T-1, F-1)
        
        # encoder
        x1, e1 = self.encoder1(x0)
        x2, e2 = self.encoder2(x1)
        x3, e3 = self.encoder3(x2)
        x4, e4 = self.encoder4(x3)
        x5, e5 = self.encoder5(x4)
        x6, e6 = self.encoder6(x5)
        
        # bottleneck
        x6 = F.dropout2d(self.bottle1(x6), p=0.1, training=self.training)
        x6 = F.dropout2d(self.bottle2(x6), p=0.1, training=self.training)
        x6 = F.dropout2d(self.bottle3(x6), p=0.1, training=self.training)
        x6 = F.dropout2d(self.bottle4(x6), p=0.1, training=self.training)
        
        # decoder
        x5 = self.decoder6(x6, e6)
        x4 = self.decoder5(x5, e5)
        x3 = self.decoder4(x4, e4)
        x2 = self.decoder3(x3, e3)
        x1 = self.decoder2(x2, e2)
        x0 = self.decoder1(x1, e1)
        
        # postnet
        x0 = sum([x0] + [f(x0) for f in self.postnet])
        x0 = F.leaky_relu(x0)
        
        # output
        mx_hat = torch.sigmoid(self.mask(x0).squeeze(1).transpose(1,2)) * my # (B, F-1, T-1)
        P = F.normalize(self.twiddle(x0).transpose(1,3), p=2.0, dim=-1, eps=1e-7) # (B, F-1, T-1, 2)
        real = (py[..., 0] * P[..., 0] - py[..., 1] * P[..., 1]).contiguous()
        imag = (py[..., 1] * P[..., 0] + py[..., 0] * P[..., 1]).contiguous()
        sx_hat = torch.stack((real*mx_hat, imag*mx_hat), dim=-1) #(B, F-1, T-1, 2)        
        mx_hat = torch.norm(sx_hat, p=2, dim=-1) # (B, F-1, T-1)
        
        # istft
        x_hat = self._istft(F.pad(sx_hat, (0, 0, 0, 0, 0, 1))) # (B, t-hop_size)
        
        return x_hat, (mx, mx_hat), (sx, sx_hat)
    
    @torch.no_grad()
    def infer(self, y, tta=False):
        # y: mixture signal, shape=(B, t), t=audio length
        
        # valid length
        orig_length = y.size(-1)
        segment_length = np.prod(self.downsample_factors) * self.hop_size
        pad_length = (segment_length - self.hop_size) - (orig_length % segment_length)
        if pad_length < 0: pad_length += segment_length
        pad0 = pad_length // 2
        pad1 = pad_length - pad0
        y = F.pad(y, (pad0, pad1))
        
        # stft
        sy = self._stft(y.to(self.window.device))[:, :-1] # (B, F-1, T, 2)
        assert sy.size(2) % np.prod(self.downsample_factors) == 0
        
        # magnitude and phase
        my = torch.norm(sy, p=2, dim=-1) # (B, F-1, T)
        py = F.normalize(sy, p=2, dim=-1, eps=1e-7) # (B, F-1, T, 2)
        
        # prenet
        x0 = self.prenet(sy.transpose(1,3)) # <- (B, 2, T, F-1)
        
        # encoder
        x1, e1 = self.encoder1(x0)
        x2, e2 = self.encoder2(x1)
        x3, e3 = self.encoder3(x2)
        x4, e4 = self.encoder4(x3)
        x5, e5 = self.encoder5(x4)
        x6, e6 = self.encoder6(x5)
        
        # bottleneck
        x6 = F.dropout2d(self.bottle1(x6), p=0.1, training=tta)
        x6 = F.dropout2d(self.bottle2(x6), p=0.1, training=tta)
        x6 = F.dropout2d(self.bottle3(x6), p=0.1, training=tta)
        x6 = F.dropout2d(self.bottle4(x6), p=0.1, training=tta)
        
        # decoder
        x5 = self.decoder6(x6, e6)
        x4 = self.decoder5(x5, e5)
        x3 = self.decoder4(x4, e4)
        x2 = self.decoder3(x3, e3)
        x1 = self.decoder2(x2, e2)
        x0 = self.decoder1(x1, e1)
        
        # postnet
        x0 = sum([x0] + [f(x0) for f in self.postnet])
        x0 = F.leaky_relu(x0)
        
        # output
        mx_hat = torch.sigmoid(self.mask(x0).squeeze(1).transpose(1,2)) * my # (B, F-1, T-1)
        P = F.normalize(self.twiddle(x0).transpose(1,3), p=2.0, dim=-1, eps=1e-7) # (B, F-1, T-1, 2)
        real = (py[..., 0] * P[..., 0] - py[..., 1] * P[..., 1]).contiguous()
        imag = (py[..., 1] * P[..., 0] + py[..., 0] * P[..., 1]).contiguous()
        sx_hat = torch.stack((real*mx_hat, imag*mx_hat), dim=-1) #(B, F-1, T-1, 2)        
        
        # istft
        x_hat = self._istft(F.pad(sx_hat, (0, 0, 0, 0, 0, 1))) # (B, T*hop_size)
        
        pad1 = None if pad1 == 0 else -pad1
        x_hat = x_hat[..., pad0:pad1]
        assert x_hat.size(-1) == orig_length
        
        return x_hat


