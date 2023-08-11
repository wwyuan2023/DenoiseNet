# -*- coding: utf-8 -*-

# Copyright 2020 Alibaba Cloud
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VectorQuantizer module."""
    
    def __init__(self, n_class, n_dim, interval=1000):
        """Initialize VectorQuantizer module."""
        super(VectorQuantizer, self).__init__()
        self.n_class = n_class
        self.n_dim = n_dim
        self.interval = interval

        self.codebook = nn.Embedding(n_class, n_dim)
        self.codebook.weight.data.uniform_(-1., 1.)
        
        self.register_buffer('stats', torch.zeros(n_class, dtype=torch.float), persistent=False)
        self.register_buffer('index', torch.arange(n_class, dtype=torch.long), persistent=False)
    
    def _scheme_threshold(self, steps):
        if steps is None or steps <= 0: return 1e+6
        
        if steps < 1e+4: threshold = 1e+4
        if steps < 5e+4: threshold = 1e+5
        elif steps < 1e+5: threshold = 5e+5
        elif steps < 2e+5: threshold = 1e+6
        elif steps < 5e+5: threshold = 2e+6
        elif steps < 1e+6: threshold = 5e+6
        else: threshold = 1e+7
        
        return threshold
    
    def forward(self, x, steps=0):
        """Calculate forward propagation.
        
        Args:
            x (Tensor): Input tensor (B, T, in_channels).
            
        Returns:
            quanted     quantized `x`.
            entropy     the exponent formate of entropy information.
        """
        B, T, C = x.size()
        N = self.n_class
        
        x_flat = x.detach().view(-1,C) # (B*T, C)
        weight = self.codebook.weight.detach()        # (N, C)
        
        # calculate distances
        input2 = torch.sum(x_flat**2, dim=1, keepdim=True)
        embed2 = torch.sum(weight**2, dim=1)
        distance = torch.addmm(embed2 + input2, x_flat, weight.t(), alpha=-2.0, beta=1.0)
        
        # encoding
        encoding_idx = torch.argmin(distance, dim=1, keepdim=True) # (B*T, 1)
        encoding_zero = torch.zeros(B*T, N, device=x.device, dtype=x.dtype)
        encodings = encoding_zero.scatter(1, encoding_idx, 1) # one-hot, (B*T, N)
        
        # full-filling codebook
        if self.training and steps > 0:
            self.stats += torch.sum(encodings, dim=0) # (N,)
            if steps % self.interval == 0: # and self.stats.sum().item() > self._scheme_threshold(steps):
                unused = (self.stats<1).bool() # unused mask, (N,)
                num_unused = torch.sum(unused)
                logging.info(f"VQ layer(n_class={self.n_class}, n_dim={self.n_dim}): "
                    "steps={}, sum={}, unused={}, unused indices={}".format(steps, self.stats.sum().item(), num_unused.item(), self.index[unused].cpu().numpy())
                )
                if num_unused > 0:
                    one_unused = self.index[unused][-1] # get one unused
                    props = num_unused / float(self.n_class)
                    props = max(0.05, min(0.2, props))
                    bm = torch.distributions.binomial.Binomial(1, probs=props)
                    mask = bm.sample(encoding_idx.size()) # 0/1, (B*T,)
                    encoding_idx[mask>0] = one_unused
                    encodings = encoding_zero.scatter(1, encoding_idx, 1)
                self.stats.fill_(0)
        
        # quantize and unflatten
        quanted = torch.matmul(encodings, self.codebook.weight).view(B,T,C)
        
        # entropy info.
        avg_probs = torch.mean(encodings, dim=0) # (N,)
        entropy = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # loss
        diff = (x - quanted).pow(2).mean()
        
        # straight-through while training
        if self.training:
            quanted_ = x + (quanted - x).detach()
            quanted = quanted * 0.25 + quanted_ * 0.75
        
        return encoding_idx, quanted, diff, entropy




