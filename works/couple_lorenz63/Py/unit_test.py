#!/usr/bin/env python

import sys
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
  yo = np.array(\
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
  ptb = 1.0e-9
  step_verif = 10

  # initialization (verification t0 -> t1)
  x_t0 = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(STEPS):
    x_t0 = timestep(x_t0, DT)
  x_t1 = np.copy(x_t0)
  for i in range(step_verif):
    x_t1 = timestep(x_t1, DT)

  # get tangent linear
  m = finite_time_tangent(x_t0, DT/1.0, step_verif*1)
  # m = finite_time_tangent_using_nonlinear(x_t0, DT/1.0, step_verif*1)

  sum_sq_diff = 0.0
  for i in range(DIMM):
    # nonlinear perturbation
    x_t0_ptb = np.copy(x_t0)
    x_t0_ptb[i] = x_t0[i] + ptb
    print(i)
    x_t1_ptb = np.copy(x_t0_ptb)
    for j in range(step_verif):
      x_t1_ptb = timestep(x_t1_ptb, DT)

    print("nonlinear:")
    print((x_t1_ptb - x_t1) / ptb)
    print("tangent linear:")
    print(m[:,i].flatten())
    print("diff (NL - TL):")
    diff = (x_t1_ptb - x_t1) / ptb - m[:,i].flatten()
    print(diff)
    sum_sq_diff = sum_sq_diff + np.sum(diff ** 2)

  print("total RMS of diff:")
  print(np.sqrt(sum_sq_diff))
  return 0

def test_tangent_sv():
  step_verif = 10

  x_t0 = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(STEPS):
    x_t0 = timestep(x_t0, DT)
  m_finite = finite_time_tangent(x_t0, DT/4.0, step_verif*4)
  mt_finite = finite_time_tangent(x_t0, DT/4.0, step_verif*4).T
  eig_vals2, eig_vects2 = np.linalg.eig(m_finite * mt_finite)
  eig_vals3, eig_vects3 = np.linalg.eig(m_finite)
  print("SV growth rates:")
  print(eig_vals2)
  print("LV growth rates:")
  print(eig_vals3)
  print("covariant LVs:")
  print(eig_vects3)
  print("M:")
  print(m_finite)
  print("M.T:")
  print(mt_finite)
  return 0

def test_cost_function_grad():
  if DIMM != 3:
    sys.exit("Set DIMM = 3 for test_cost_function_grad")

  aint = 25
  fcst = np.array([-5.83559367, -6.1021729, 23.42678068])
  anl = np.copy(fcst)
  h = geth([1,1,1])
  r = getr()
  yo = np.array([[-8.27064106], [-1.06064404], [34.80718227]])
  eps = 1.0

  twoj = fdvar_2j(anl, fcst, h, r, yo, aint, 0, DIMM)
  twoj_grad = np.zeros(DIMM)
  for i in range(DIMM):
    anl = np.copy(fcst)
    anl[i] += eps
    twoj_grad[i] = (fdvar_2j(anl, fcst, h, r, yo, aint, 0, DIMM) - twoj) / eps
  print(twoj_grad)

  twoj_grad_anl = fdvar_2j_deriv(anl, fcst, h, r, yo, aint, 0, DIMM)
  print(twoj_grad_anl)

  return 0

test_tangent_model()
# test_cost_function_grad()
