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

  # for t-continuity, align
  for i in range(DIMM):
    inner_prod = np.dot(q[:,i].T, lv[:,i])
    if (inner_prod < 0):
      q[:,i] *= -1.0

  lv_orth = q * eps
  le = np.log(eigvals / eps)
  return lv_orth, le

def vector_common(blv, flv, k, eps):
  # blv     <- np.array[DIMM,k]        : backward Lyapunov vectors (column)
  # flv     <- np.array[DIMM,DIMM-k+1] : forward Lyapunov vectors (column)
  # k       <- int                     : [1,DIMM)
  # eps     <- float
  # return  -> np.array[DIMM]          : k+1 th characteristic Lyapunov vectors
  ab = np.empty((DIMM,DIMM+1))
  ab[:,:k+1] = blv[:,:]
  ab[:,k+1:] = - flv[:,:]
  coefs = nullspace(ab)
  clv = np.dot(ab[:,:k+1], coefs[:k+1,np.newaxis])[:,0]
  clv_len = np.sqrt(np.sum(clv**2))
  return clv / clv_len * eps

def nullspace(a):
  # refer to 658Ep19
  u, s, vh = np.linalg.svd(a)
  return vh.T[:,-1]
