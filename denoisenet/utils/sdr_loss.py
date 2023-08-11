# -*- coding: utf-8 -*-


"""Calculate SI-SDR"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_sdr(x_hat, x, eps=1e-7):
    # x_hat: generated, (B, C, T)
    # x: groundturth, (B, C, T)
    num = torch.sum(torch.square(x), dim=(1,2))
    den = torch.sum(torch.square(x - x_hat), dim=(1,2))
    num += eps
    den += eps
    scores = 10 * torch.log10(num / den) # (B,)
    return scores.mean()


def si_sdr(x_hat, x):
    # x/x_hat: (..., 2), where `x` is groundturth and `x_hat` is prediction.
    # SI-SDR = 10*log10(|xt|^2/|xe|^2), xt=<x_hat,x>*x/|x|^2, xe=x_hat-xt
    xt = torch.sum(x_hat*x, dim=-1, keepdim=True) * x / torch.clamp(torch.norm(x, dim=-1, keepdim=True)**2, min=1e-5)
    xe = x_hat - xt
    r = xt**2 / torch.clamp(xe**2, min=1e-5)
    return 10 * torch.log10(torch.clamp(r, min=1e-10)) # (...,)


class SISDRLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction # mean, sum or None
    
    def forward(self, x_hat, x):
        if self.reduction is None:
            return -1. * si_sdr(x_hat, x)
        return -1. * getattr(torch, self.reduction)(si_sdr(x_hat, x))
