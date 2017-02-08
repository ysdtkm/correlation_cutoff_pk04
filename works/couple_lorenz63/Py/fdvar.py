#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin, fmin_bfgs
from const import *
from tdvar import *
from model import *

def fdvar(fcst_0, h, r, yo, aint):
  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time
  anl_0 = fcst_0
  print_ndarray(anl_0, "anl_0")
  print_ndarray(yo, "yo")
  try:
    anl_0 = fmin_bfgs(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint))
  except:
    print("Method fmin_bfgs failed to converge. Use fmin for this step instead.")
    anl_0 = fmin(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint))
  anl_1 = anl_0
  for i in range(0, aint):
    anl_1 = timestep(anl_1, DT)
  return anl_1.T

def fdvar_2j(anl_0, fcst_0, h, r, yo, aint):
  b = 0.6 * 1.2 * tdvar_b()
  anl_tmp = anl_0
  anl_0 = np.matrix(anl_tmp).T
  anl_1_ar = anl_0.A.flatten()
  for i in range(0, aint):
    anl_1_ar = timestep(anl_1_ar, DT)
  anl_1 = np.matrix(anl_1_ar).T
  twoj = (anl_0 - fcst_0).T * b.I * (anl_0 - fcst_0) + \
       (h * anl_1 - yo).T * r.I * (h * anl_1 - yo)
  return twoj.A

def print_ndarray(nar, name):
  n = nar.shape[0]
  m = nar.shape[1]
  print("%s:" % name)
  for i in range(n):
    for j in range(m):
      print(float(nar[i,j]))
  print("")
