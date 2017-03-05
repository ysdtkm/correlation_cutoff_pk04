#!/usr/bin/env python

import os
import numpy as np
from const import *
from model import *

def calc_blv_fsv(all_true):
  eps = 1.0e-9
  orth_int = 1

  blv = np.random.normal(0.0, eps, (DIMM, DIMM))
  blv, ble = orth_norm_vectors(blv, eps)
  all_blv = np.zeros((STEPS, DIMM, DIMM))
  all_ble = np.zeros((STEPS, DIMM))

  for i in range(1, STEPS):
    true = all_true[i-1,:]
    m = finite_time_tangent_using_nonlinear(true, DT, 1)
    blv = np.dot(m, blv)
    if (i % orth_int == 0):
      # (ble[:] / DT) is (orth_int * actual LEs).
      # For window without orthonormalization, LEs are zero.
      # Thus long-term mean wil be actual LEs.
      blv, ble = orth_norm_vectors(blv, eps)
      all_ble[i,:] = ble[:] / DT
    all_blv[i,:,:] = blv[:,:] / eps

  all_blv.tofile("data/blv.bin")
  all_ble.tofile("data/ble.bin")
  return all_blv, all_ble

def calc_flv_isv(all_true):
  eps = 1.0e-9
  orth_int = 1

  flv = np.random.normal(0.0, eps, (DIMM, DIMM))
  flv, fle = orth_norm_vectors(flv, eps)
  all_flv = np.empty((STEPS, DIMM, DIMM))
  all_fle = np.zeros((STEPS, DIMM))

  for i in range(STEPS, 0, -1):
    true = all_true[i-1,:]
    m = finite_time_tangent_using_nonlinear(true, DT, 1)
    flv = np.dot(m.T, flv)
    if (i % orth_int == 0):
      flv, fle = orth_norm_vectors(flv, eps)
      all_fle[i-1,:] = fle[:] / DT
    all_flv[i-1,:,:] = flv[:,:] / eps
  all_flv.tofile("data/flv.bin")
  all_fle.tofile("data/fle.bin")
  return all_flv, all_fle

def calc_clv(all_true, all_blv, all_flv):
  all_clv = np.empty((STEPS, DIMM, DIMM))

  for i in range(1, STEPS):
    for k in range(0, DIMM):
      all_clv[i,:,k] = vector_common(all_blv[i,:,:k+1], all_flv[i,:,k:], k, 1.0)

    # directional continuity
    if (i >= 2):
      m = finite_time_tangent_using_nonlinear(all_true[i-1,:], DT, 1)
      for k in range(0, DIMM):
        clv_approx = np.dot(m, all_clv[i-1,:,k,np.newaxis]).flatten()
        if (np.dot(clv_approx, all_clv[i,:,k]) < 0):
          all_clv[i,:,k] *= -1

  all_clv.tofile("data/clv.bin")
  return all_clv

def write_lyapunov_exponents(all_ble, all_fle, all_clv):
  f = open("data/lyapunov.txt", "w")
  f.write("backward LEs:\n")
  f.write(str_vector(np.mean(all_ble[STEPS//2:,:], axis=0)) + "\n")
  f.write("forward LEs:\n")
  f.write(str_vector(np.mean(all_fle[STEPS//2:,:], axis=0)) + "\n")
  f.write("CLV RMS (column: LVs, row: model grid):\n")
  for i in range(DIMM):
    f.write(str_vector(np.mean(all_clv[STEPS//2:,i,:]**2, axis=0)) + "\n")
  f.close()
  os.system("cat data/lyapunov.txt")
  return 0

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

def str_vector(arr):
  n = len(arr)
  st = ""
  for i in range(n):
    st += "%11g, " % arr[i]
  return st[:-2]

