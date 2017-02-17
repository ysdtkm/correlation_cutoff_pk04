#!/usr/bin/env python

import numpy as np
from const import *

def orth_norm_vectors(lv, eps):
  # lv     <- np.array[DIMM,DIMM] : Lyapunov vectors (column)
  # eps    <- float               : small size to normalize LVs
  # return -> np.array[DIMM,DIMM] : orthonormalized LVs in descending order
  # return -> np.array[DIMM]      : ordered Lyapunov Exponents

  q, r = np.linalg.qr(lv)

  le = np.zeros(DIMM)
  eigvals = np.abs(np.diag(r))
  order_dsc = np.argsort(eigvals)[::-1]
  for i in range(DIMM):
    lv[:,i] = q[:,order_dsc[i]]
    lv[:,i] *= eps
    le[i] = eigvals[order_dsc[i]] / eps
  return lv, le
