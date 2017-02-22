#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin, fmin_bfgs
from const import *
from tdvar import *
from model import *

def fdvar(fcst_0, h, r, yo, aint):
  # fcst_0 <- np.array[DIMM]       : first guess at beginning of window
  # h      <- np.array[DIMO, DIMM] : observation operator
  # r      <- np.array[DIMO, DIMO] : observation error covariance
  # yo     <- np.array[DIMO, 1]    : observation
  # aint   <- int                  : assimilation interval
  # return -> np.array[DIMM]       : assimilated field

  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time
  anl_0 = np.copy(fcst_0)
  try:
    anl_0 = fmin_bfgs(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint))
  except:
    print("Method fmin_bfgs failed to converge. Use fmin for this step instead.")
    anl_0 = fmin(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint))
  anl_1 = np.copy(anl_0)
  for i in range(0, aint):
    anl_1 = timestep(anl_1, DT)
  return anl_1.T

def fdvar_2j(anl_0_nda, fcst_0_nda, h_nda, r_nda, yo_nda, aint):
  # anl_0_nda  <- np.array[DIMM]       : temporary analysis field
  # fcst_0_nda <- np.array[DIMM]       : first guess field
  # h_nda      <- np.array[DIMO, DIMM] : observation operator
  # r_nda      <- np.array[DIMO, DIMO] : observation error covariance
  # yo_nda     <- np.array[DIMO, 1]    : observation
  # aint       <- int                  : assimilation interval
  # return     -> float                : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(0.6 * 1.2 * tdvar_b())
  anl_0  = np.asmatrix(anl_0_nda).T
  fcst_0 = np.asmatrix(fcst_0_nda).T

  anl_1_nda = np.copy(anl_0_nda)
  for i in range(0, aint):
    anl_1_nda = timestep(anl_1_nda, DT)

  # all array-like objects below are np.matrix
  anl_1 = np.matrix(anl_1_nda).T
  twoj = (anl_0 - fcst_0).T * b.I * (anl_0 - fcst_0) + \
         (h * anl_1 - yo).T * r.I * (h * anl_1 - yo)
  return twoj.A

