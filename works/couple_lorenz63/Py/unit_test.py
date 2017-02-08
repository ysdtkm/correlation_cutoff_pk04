#!/usr/bin/env python

import numpy as np
from const import *
from model import *
from fdvar import *

def test_fdvar_overflow():
  exp = EXPLIST[1]
  h = geth(exp["diag"])
  r = getr()
  fcst_0 = np.array( \
    [[-5.443006274201698], \
    [-0.8043972209074476], \
    [29.448353421328868], \
    [-6.875648898446565], \
    [-10.746162351175121], \
    [45.01895925307126], \
    [-40.33122967882807], \
    [-92.74048173257268], \
    [81.4746920842174]] \
  )
  yo = np.matrix(\
    [[-2.9312942806408193], \
    [-5.972428159803157], \
    [14.373259247649628], \
    [0.14974816307418903], \
    [-3.393558982949476], \
    [47.920238909038034], \
    [-39.34468967151646], \
    [-12.512758026367278], \
    [117.90228464480583]] \
  )
  aint = exp["aint"]

  fdvar(fcst_0, h, r, yo, aint)
  return 0

def test_tangent_model():
  x_t0 = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(STEPS):
    x_t0 = timestep(x_t0, DT)
  x_t1 = timestep(x_t0, DT)
  ptb = 1.0e-9
  m = tangent_linear(x_t0, DT)
  sum_sq_diff = 0.0

  for i in range(DIMM):
    x_t0_ptb = np.copy(x_t0)
    x_t0_ptb[i] = x_t0[i] + ptb

    print(i)
    x_t1_ptb = timestep(x_t0_ptb, DT)

    print("nonlinear:")
    print((x_t1_ptb - x_t1) / ptb)
    print("tangent linear:")
    print(m[:,i].A.flatten())
    print("diff (NL - TL):")
    diff = (x_t1_ptb - x_t1) / ptb - m[:,i].A.flatten()
    print(diff)
    sum_sq_diff = sum_sq_diff + np.sum(diff ** 2)

  print("total RMS of diff:")
  print(np.sqrt(sum_sq_diff))
  return 0

def test_tangent_sv():
  x_t0 = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(STEPS):
    x_t0 = timestep(x_t0, DT)
  m = tangent_linear(x_t0, DT)
  eig_vals1, eig_vects1 = np.linalg.eig(m * m.T)
  eig_vals2, eig_vects2 = np.linalg.eig(m.T * m)
  eig_vals3, eig_vects3 = np.linalg.eig(m)
  print("SV growth rates:")
  print(eig_vals1)
  print(eig_vals2)
  print("initial SVs:")
  print(eig_vects1)
  print("final SVs:")
  print(eig_vects2)
  print("LV growth rates:")
  print(eig_vals3)
  print("covariant LVs:")
  print(eig_vects3)
  print("M:")
  print(m)
  print("M * M.T:")
  print(m * m.T)
  return 0

test_tangent_sv()
