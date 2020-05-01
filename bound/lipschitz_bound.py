from collections import defaultdict
from itertools import product
from functools import reduce
import numpy as np
import scipy as sp
from scipy.sparse import linalg
import logging

try:
  import torch
  import torch.nn as nn
except:
  pass


class LipschitzBound:

  def __init__(self, kernel, padding, sample=50, backend='numpy'):

    self.kernels = kernels
    self.padding = padding
    self.sample = sample
    self.backend = backend

    # verify the kernels is square
    if not kernel.shape[-1] == kernel.shape[-2]:
      raise ValueError("The last 2 dim of the kernel must be equal.")
    # verify if all kernels have odd shape
    if not kernel.shape[-1] % 2 == 1:
      raise ValueError("The dimension of the kernel must be odd.")

    if isinstance(self.kernel, np.ndarray):
      self.compute_with_torch = False
      self.compute = self._compute_from_numpy
    elif isinstance(self.kernel, torch.Tensor):
      self.compute_with_torch = True
      self.cuda = self.kernel.is_cuda
      self.compute = self._compute_from_torch
    else:
      raise ValueError('kernel type not recognized.')

    # define search space
    x = np.linspace(0, 2*np.pi, num=self.sample)
    w = np.array(list(product(x, x)))
    self.w0 = w[:, 0].reshape(-1, 1)
    self.w1 = w[:, 1].reshape(-1, 1)

    if self.compute_with_torch:
      self.w0 = torch.FloatTensor(np.float32(self.w0))
      self.w1 = torch.FloatTensor(np.float32(self.w1))
      if self.cuda:
        self.w0 = self.w0.cuda()
        self.w1 = self.w1.cuda()

  def _get_sample(self, *args, **kwargs):
    if self.compute_with_torch:
      return self._get_sample_for_torch(*args, **kwargs)
    return self._get_sample_for_numpy(*args, **kwargs)

  def _get_sample_for_numpy(self, ksize, pad):
    p_index = np.arange(-ksize + 1., 1.) + pad
    H0 = 1j * np.tile(p_index, ksize).reshape(ksize, ksize).T.reshape(-1)
    H1 = 1j * np.tile(p_index, ksize)
    return np.exp(self.w0 * H0 + self.w1 * H1).T

  def _get_sample_for_torch(self, ksize, pad):
    p_index = torch.arange(-ksize + 1.0, 1.0) + pad
    H0 = p_index.repeat(ksize).reshape(ksize, ksize).T.reshape(-1)
    H1 = p_index.repeat(ksize)
    if self.cuda:
      H0 = H0.cuda()
      H1 = H1.cuda()
    real = torch.cos(self.w0 * H0 + self.w1 * H1).T
    imag = torch.sin(self.w0 * H0 + self.w1 * H1).T
    return real, imag

  def _compute_from_numpy(self):
    """Compute the LipGrid Algorithm."""
    kernel = self.kernel
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = np.transpose(kernel, axes=[1, 0, 2, 3])
      cout, cin = cin, cout
    ker = kernel.reshape(cout, cin, -1)[..., np.newaxis]
    poly = (ker * self._get_sample(ksize, pad)).sum(axis=2)
    poly = np.square(np.abs(poly)).sum(axis=1)
    sv_max = np.sqrt(poly.max(axis=-1).sum())
    return sv_max

  def _compute_from_torch(self):
    """Compute the LipGrid Algo with Torch"""
    # only one kernel at a time with torch
    kernel = self.kernel
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = torch.transpose(kernel, 0, 1)
      cout, cin = cin, cout
    ker = kernel.reshape(cout, cin, -1)[..., np.newaxis]
    real, imag = self._get_sample(ksize, pad)
    poly_real = (ker * real).sum(axis=2)
    poly_imag = (ker * imag).sum(axis=2)
    poly = torch.mul(poly_real, poly_real) + \
        torch.mul(poly_imag, poly_imag)
    poly = poly.sum(axis=1)
    sv_max = torch.sqrt(poly.max(axis=-1)[0].sum())
    return sv_max
 



