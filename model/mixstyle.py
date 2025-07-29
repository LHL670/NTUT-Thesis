#
# File: models/mixstyle.py (FINAL ROBUST VERSION)
# Description: This version adds a safeguard to handle batches with size less than 2.
#

import torch
import torch.nn as nn
import random

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): The probability of applying MixStyle.
          alpha (float): The alpha parameter for the Beta distribution.
          eps (float): A small value to avoid division by zero.
          mix (str): The mixing strategy, can be 'random' or 'crossdomain'.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.mix = mix
        self._activated = True

    def set_activation_status(self, status=True):
        """Enable or disable MixStyle."""
        self._activated = status

    def forward(self, x):
        if not self.training or not self._activated or random.random() > self.p:
            return x

        B = x.size(0)

        # ========== SAFEGUARD ADDED HERE ==========
        # If batch size is less than 2, mixing is not possible.
        if B < 2:
            return x
        # ==========================================

        # 根據選擇的策略決定如何排列風格
        if self.mix == 'random':
            # 隨機混合：從整個批次中隨機打亂
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # 跨領域混合：將批次分成兩半，並交換它們的風格
            perm = torch.arange(B - 1, -1, -1)
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])] # Use perm_b.shape[0] for robustness
            perm_a = perm_a[torch.randperm(perm_a.shape[0])] # Use perm_a.shape[0] for robustness
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError(f"Mix strategy '{self.mix}' is not implemented.")


        # 產生混合權重 
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        # 核心混合邏輯 
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        mu2 = mu[perm]
        sig2 = sig[perm]

        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
