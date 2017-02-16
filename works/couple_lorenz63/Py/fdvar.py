#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin, fmin_bfgs
from const import *
from tdvar import *
from model import *

def fdvar(fcst_0, h, r, yo, aint):
  # fcst_0 <- np.array[DIMM]        : first guess at beginning of window
  # h      <- np.matrix[DIMO, DIMM] : observation operator
  # r      <- np.matrix[DIMO, DIMO] : observation error covariance
  # yo     <- np.matrix[DIMO, 1]    : observation
  # aint   <- int                   : assimilation interval
  # return -> np.array[DIMM]        : assimilated field

  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time
  anl_0 = fcst_0
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
  # anl_0  <- np.array[DIMM]        : temporary analysis field
  # fcst_0 <- np.array[DIMM]        : first guess field
  # h      <- np.matrix[DIMO, DIMM] : observation operator
  # r      <- np.matrix[DIMO, DIMO] : observation error covariance
  # yo     <- np.matrix[DIMO, 1]    : observation
  # aint   <- int                   : assimilation interval
  # return -> float                 : cost function 2J

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

