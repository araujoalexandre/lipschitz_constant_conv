from datetime import datetime
import numpy as np
import scipy as sp
from scipy.sparse import linalg

import torch
import torchvision.models as models

from block_matrices.ops import DoublyBlockToeplitzFromKernel
from lipschitz_bound import LipschitzBound


def test_single_kernel(kernel=None, kshape=None, image_size=None,
                       padding=None, sample=50, dump_kernel=None):
  if kernel is None:
    cout, cin, ksize, ksize = kshape
    kernel = np.random.randn(cout, cin, ksize, ksize)
    kernel = torch.FloatTensor(np.float32(kernel))
    # kernel = kernel.cuda()
  cout, cin, ksize, ksize = kernel.shape
  A = DoublyBlockToeplitzFromKernel(
    image_size, kernel.cpu().numpy(), padding=padding, return_sparse=True).generate()
  sv_max = linalg.svds(
    A, k=1, which='LM', return_singular_vectors=False)[0]

  lb = LipschitzBound(kernel, padding, sample=sample) 
  start = datetime.now()
  sv_bound = lb.compute()
  end = (datetime.now() - start).total_seconds()

  msg = ("Single kernel {}x{}x{}x{}: "
         "sv_max: {:.5f}, sv_bound: {:.5f}, time: {}")
  print(msg.format(cout, cin, ksize, ksize, sv_max, sv_bound, end))


if __name__ == '__main__':

  test_single_kernel(kshape=(1, 1, 3, 3), image_size=29, padding=1, sample=300) 

  # for ksize in [3, 5, 7, 9, 11]:
  #   for padding in [0, 1, 2, 3, 4]:
  #     kshape = [1, 1, ksize, ksize]
  #     test_single_kernel(kshape=kshape, image_size=60,
  #                        padding=padding, sample=300)
  #
  # for cout in [6, 9, 12]:
  #   for cin in [6, 9, 12]:
  #     kshape = [cout, cin, 3, 3]
  #     test_single_kernel(kshape=kshape, image_size=60,
  #                        padding=1, sample=200)

  # shapes = [(4, 4, 3, 3), (20, 20, 3, 3)]
  # for kshape in shapes:
  #   test_single_kernel(kshape=kshape, image_size=60,
  #                      padding=1, sample=200)


