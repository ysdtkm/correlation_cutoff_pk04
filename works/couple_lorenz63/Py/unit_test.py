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
  dx_t0 = tendency(x_t0)
  diff = 1.0e-9
  m = tangent_linear(x_t0)

  for i in range(DIMM):
    x_t0_ptb = np.copy(x_t0)
    x_t0_ptb[i] = x_t0[i] + diff

    print(i)
    dx_t0_ptb = tendency(x_t0_ptb)

    print("nonlinear:")
    print((dx_t0_ptb - dx_t0) / diff)
    print("tangent linear:")
    print(m[:,i].A.flatten())
    print("diff (NL - TL):")
    print((dx_t0_ptb - dx_t0) / diff - m[:,i].A.flatten())

  return 0

test_tangent_model()
