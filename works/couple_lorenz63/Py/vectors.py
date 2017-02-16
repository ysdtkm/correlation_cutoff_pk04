#!/usr/bin/env python

import numpy as np
from const import *

def orth_norm_vectors(lv, eps):
  # lv     <- np.array[DIMM,DIMM] : Lyapunov vectors
  # eps    <- float               : small size to normalize LVs
  # return -> np.array[DIMM,DIMM] : orthonormalized LVs in descending order

  # need to check if the column vectors of q are unit length
  q, r = np.linalg.qr(lv)
  lv0 = np.copy(lv)

  eigvals = np.diag(r)
  order = np.argsort(eigvals)
  for i in range(DIMM):
    lv[i,:] = lv0[order[i],:]
    norm = np.sqrt(np.sum(np.power(lv[i,:], 2)))
    lv[i,:] *= eps / norm
  return lv
