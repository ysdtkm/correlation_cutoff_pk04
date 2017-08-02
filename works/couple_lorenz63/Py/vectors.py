#!/usr/bin/env python

import os
import numpy as np
from const import *
import model
import better_exceptions

def calc_blv(all_true):
  # all_true <- np.array[STEPS, DIMM]
  # all_blv  -> np.array[STEPS, DIMM, DIMM] : column backward lvs
  # all_ble  -> np.array[STEPS, DIMM]       : backward lyapunov exponents, zero for the first step

  orth_int = 1

  blv = np.random.randn(DIMM, DIMM)
  blv, ble = orth_norm_vectors(blv)
  all_blv = np.zeros((STEPS, DIMM, DIMM))
  all_ble = np.zeros((STEPS, DIMM))

  for i in range(1, STEPS):
    true = all_true[i-1,:].copy()
    m = model.finite_time_tangent_using_nonlinear(true, DT, 1)
    blv = np.dot(m, blv)
    if (i % orth_int == 0):
      blv, ble = orth_norm_vectors(blv)
      all_ble[i,:] = ble[:] / DT
    all_blv[i,:,:] = blv[:,:]

  all_blv.tofile("data/blv.bin")
  all_ble.tofile("data/ble.bin")
  return all_blv, all_ble

def calc_flv(all_true):
  # all_true <- np.array[STEPS, DIMM]
  # all_flv  -> np.array[STEPS, DIMM, DIMM] : column forward lvs
  # all_fle  -> np.array[STEPS, DIMM]       : forward lyapunov exponents, zero for the last step

  orth_int = 1

  flv = np.random.randn(DIMM, DIMM)
  flv, fle = orth_norm_vectors(flv)
  all_flv = np.zeros((STEPS, DIMM, DIMM))
  all_fle = np.zeros((STEPS, DIMM))

  for i in range(STEPS, 0, -1):
    true = all_true[i-1,:].copy()
    m = model.finite_time_tangent_using_nonlinear(true, DT, 1)
    flv = np.dot(m.T, flv)
    if (i % orth_int == 0):
      flv, fle = orth_norm_vectors(flv)
      all_fle[i-1,:] = fle[:] / DT
    all_flv[i-1,:,:] = flv[:,:]

  all_flv.tofile("data/flv.bin")
  all_fle.tofile("data/fle.bin")

  return all_flv, all_fle

def calc_clv(all_true, all_blv, all_flv):
  # all_true <- np.array[STEPS, DIMM]
  # all_blv  <- np.array[STEPS, DIMM, DIMM] : column backward lvs
  # all_flv  <- np.array[STEPS, DIMM, DIMM] : column forward lvs
  # all_clv  -> np.array[STEPS, DIMM, DIMM] : column charactetistic lvs, zero for the first and last step

  all_clv = np.zeros((STEPS, DIMM, DIMM))

  for i in range(1, STEPS-1):
    for k in range(0, DIMM):
      all_clv[i,:,k] = vector_common(all_blv[i,:,:k+1], all_flv[i,:,k:], k)

    # directional continuity
    if (i >= 2):
      m = model.finite_time_tangent_using_nonlinear(all_true[i-1,:], DT, 1)
      for k in range(0, DIMM):
        clv_approx = np.dot(m, all_clv[i-1,:,k,np.newaxis]).flatten()
        if (np.dot(clv_approx, all_clv[i,:,k]) < 0):
          all_clv[i,:,k] *= -1

  all_clv.tofile("data/clv.bin")
  return all_clv

def calc_fsv(all_true):
  ### refer to 658Ep15 about exponents
  # all_true <- np.array[STEPS, DIMM]
  # all_fsv  -> np.array[STEPS, DIMM, DIMM] : column final SVs, zero for the first (window) steps
  # all_fse  -> np.array[STEPS, DIMM]       : Singular exponents (1/window) * ln(sigma)

  all_fsv = np.zeros((STEPS, DIMM, DIMM))
  all_fse = np.zeros((STEPS, DIMM))
  window = 1
  for i in range(window, STEPS):
    true = all_true[i-window,:].copy()
    m = model.finite_time_tangent_using_nonlinear(true, DT, window)
    u, s, vh = np.linalg.svd(m)
    all_fsv[i,:,:] = u[:,:]
    all_fse[i,:] = np.log(np.abs(s)) / (DT * window)
  all_fsv.tofile("data/fsv.bin")
  all_fse.tofile("data/fse.bin")
  return all_fsv

def calc_isv(all_true):
  ### refer to 658Ep15 about exponents
  # all_true <- np.array[STEPS, DIMM]
  # all_isv  -> np.array[STEPS, DIMM, DIMM] : column initial SVs, zero for the last (window) steps
  # all_ise  -> np.array[STEPS, DIMM]       : Singular exponents (1/window) * ln(sigma)

  all_isv = np.zeros((STEPS, DIMM, DIMM))
  all_ise = np.zeros((STEPS, DIMM))
  window = 1
  for i in range(STEPS, window-1, -1):
    true = all_true[i-window,:].copy()
    m = model.finite_time_tangent_using_nonlinear(true, DT, window)
    u, s, vh = np.linalg.svd(m)
    all_isv[i-window,:,:] = vh.T[:,:]
    all_ise[i-window,:] = np.log(np.abs(s)) / (DT * window)
  all_isv.tofile("data/isv.bin")
  all_ise.tofile("data/ise.bin")
  return all_isv

def write_lyapunov_exponents(all_ble, all_fle, all_clv):
  # all_ble <- np.array[STEPS, DIMM]
  # all_fle <- np.array[STEPS, DIMM]
  # all_clv <- np.array[STEPS, DIMM, DIMM]

  f = open("data/lyapunov.txt", "w")
  f.write("backward LEs:\n")
  f.write(str_vector(np.mean(all_ble[STEPS//4:(STEPS*3)//4,:], axis=0)) + "\n")
  f.write("forward LEs:\n")
  f.write(str_vector(np.mean(all_fle[STEPS//4:(STEPS*3)//4:,:], axis=0)) + "\n")
  f.write("CLV RMS (column: LVs, row: model grid):\n")
  for i in range(DIMM):
    f.write(str_vector(np.mean(all_clv[STEPS//4:(STEPS*3)//4:,i,:]**2, axis=0)) + "\n")
  f.close()
  os.system("mkdir -p image/true")
  os.system("cat data/lyapunov.txt")
  return 0

def orth_norm_vectors(lv):
  # lv     <- np.array[DIMM,DIMM] : Lyapunov vectors (column)
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

  lv_orth = q.copy()
  le = np.log(eigvals)
  return lv_orth, le

def vector_common(blv, flv, k):
  ### Note658E p19
  # blv     <- np.array[DIMM,k]        : backward Lyapunov vectors (column)
  # flv     <- np.array[DIMM,DIMM-k+1] : forward Lyapunov vectors (column)
  # k       <- int                     : [1,DIMM)
  # return  -> np.array[DIMM]          : k+1 th characteristic Lyapunov vectors (unit length)
  ab = np.empty((DIMM,DIMM+1))
  ab[:,:k+1] = blv[:,:]
  ab[:,k+1:] = - flv[:,:]
  coefs = nullspace(ab)
  clv = np.dot(ab[:,:k+1], coefs[:k+1,np.newaxis])[:,0]
  clv_len = np.sqrt(np.sum(clv**2))
  return clv / clv_len

def nullspace(a):
  # refer to 658Ep19
  # a      <- np.array[DIMM,DIMM+1]
  # return <- np.array[DIMM]
  u, s, vh = np.linalg.svd(a)
  return vh.T[:,-1].copy()

def str_vector(arr):
  # arr    <- np.array[n]
  # return -> string
  n = len(arr)
  st = ""
  for i in range(n):
    st += "%11g, " % arr[i]
  return st[:-2]

