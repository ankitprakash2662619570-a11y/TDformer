import torch
import torch.nn as nn
from PyEMD import EMD
import numpy as np

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            x = self._apply_emd_significant(x)
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def _apply_emd_significant(self, x):
        B, L, C = x.shape
        x_np = x.cpu().detach().numpy()
        emd = EMD()
        reconstructed = np.zeros_like(x_np)
        for b in range(B):
            for c in range(C):
                signal = x_np[b, :, c]
                imfs = emd.emd(signal)
                N = 100
                std = np.std(signal)
                num_imfs = len(imfs)
                energies = np.zeros((N, num_imfs))
                for i in range(N):
                    noise = np.random.normal(0, std, L)
                    noise_imfs = emd.emd(noise)
                    for j in range(min(num_imfs, len(noise_imfs))):
                        energies[i, j] = np.sum(noise_imfs[j]**2) / L
                mean_energy = np.mean(energies, axis=0)
                std_energy = np.std(energies, axis=0)
                confidence = mean_energy + 1.96 * std_energy / np.sqrt(N)
                data_energies = np.array([np.sum(imf**2) / L for imf in imfs])
                significant = data_energies > confidence[:len(data_energies)]
                reconstructed[b, :, c] = np.sum(imfs[significant], axis=0)
        return torch.from_numpy(reconstructed).to(x.device)
