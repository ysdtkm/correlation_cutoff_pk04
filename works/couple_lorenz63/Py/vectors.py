#!/usr/bin/env python

import numpy as np
from const import *

def orth_norm_vectors(lv, eps):
  # lv     <- np.array[DIMM,DIMM] : Lyapunov vectors (column)
  # eps    <- float               : small size to normalize LVs
  # return -> np.array[DIMM,DIMM] : orthonormalized (column) LVs in descending order

  # need to check if the column vectors of q are unit length
  lv0 = np.copy(lv)
  q, r = np.linalg.qr(lv)

  eigvals = np.diag(r)
  order = np.argsort(eigvals)
  for i in range(DIMM):
    lv[:,i] = q[:,order[i]]
    norm = np.sqrt(np.sum(np.power(lv[:,i], 2)))
    lv[:,i] *= eps / norm
  return lv
