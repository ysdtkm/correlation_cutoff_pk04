#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin, fmin_bfgs
from const import *
from tdvar import *
from model import *

def fdvar(fcst_0, h, r, yo, aint, i_s, i_e):
  ### here, (dimm = i_e - i_s <= DIMM) unless strongly coupled
  # fcst_0 <- np.array[dimm]       : first guess at beginning of window
  # h      <- np.array[DIMO, dimm] : observation operator
  # r      <- np.array[DIMO, DIMO] : observation error covariance
  # yo     <- np.array[DIMO, 1]    : observation
  # aint   <- int                  : assimilation interval
  # i_s    <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e    <- int
  # return -> np.array[dimm]       : assimilated field

  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time
  anl_0 = np.copy(fcst_0)
  try:
    anl_0 = fmin_bfgs(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint, i_s, i_e))
    # anl_0 = fmin_bfgs(fdvar_2j, anl_0, fprime=fdvar_2j_deriv, args=(fcst_0, h, r, yo, aint, i_s, i_e))
  except:
    print("Method fmin_bfgs failed to converge. Use fmin for this step instead.")
    anl_0 = fmin(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint, i_s, i_e), disp=False)
  anl_1 = np.copy(anl_0)
  for i in range(0, aint):
    anl_1 = timestep(anl_1, DT, i_s, i_e)
  return anl_1.T

def fdvar_2j(anl_0_nda, fcst_0_nda, h_nda, r_nda, yo_nda, aint, i_s, i_e):
  ### here, (dimm = i_e - i_s <= DIMM) unless strongly coupled
  # anl_0_nda  <- np.array[dimm]       : temporary analysis field
  # fcst_0_nda <- np.array[dimm]       : first guess field
  # h_nda      <- np.array[DIMO, dimm] : observation operator
  # r_nda      <- np.array[DIMO, DIMO] : observation error covariance
  # yo_nda     <- np.array[DIMO, 1]    : observation
  # aint       <- int                  : assimilation interval
  # i_s        <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e        <- int
  # return     -> float                : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(0.6 * 1.2 * tdvar_b()[i_s:i_e, i_s:i_e])
  anl_0  = np.asmatrix(anl_0_nda).T
  fcst_0 = np.asmatrix(fcst_0_nda).T

  anl_1_nda = np.copy(anl_0_nda)
  for i in range(0, aint):
    anl_1_nda = timestep(anl_1_nda, DT, i_s, i_e)

  # all array-like objects below are np.matrix
  anl_1 = np.matrix(anl_1_nda).T
  twoj = (anl_0 - fcst_0).T * b.I * (anl_0 - fcst_0) + \
         (h * anl_1 - yo).T * r.I * (h * anl_1 - yo)
  return twoj[0,0]

def fdvar_2j_deriv(anl_0_nda, fcst_0_nda, h_nda, r_nda, yo_nda, aint, i_s, i_e):
  ### here, (dimm = i_e - i_s <= DIMM) unless strongly coupled
  # anl_0_nda  <- np.array[dimm]       : temporary analysis field
  # fcst_0_nda <- np.array[dimm]       : first guess field
  # h_nda      <- np.array[DIMO, dimm] : observation operator
  # r_nda      <- np.array[DIMO, DIMO] : observation error covariance
  # yo_nda     <- np.array[DIMO, 1]    : observation
  # aint       <- int                  : assimilation interval
  # i_s        <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e        <- int
  # return     -> np.array[dimm]       : gradient of cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(0.6 * 1.2 * tdvar_b()[i_s:i_e, i_s:i_e])
  anl_0  = np.asmatrix(anl_0_nda).T
  fcst_0 = np.asmatrix(fcst_0_nda).T

  # todo: the method below does not handle weak/non coupled DA
  m = finite_time_tangent_using_nonlinear(fcst_0_nda, DT, aint)
  inc = anl_0 - fcst_0
  fcst_1_nda = np.copy(fcst_0_nda)
  for i in range(aint):
    fcst_1_nda = timestep(fcst_1_nda, DT)
  fcst_1 = np.asmatrix(fcst_1_nda).T
  d = yo - h * fcst_1

  j_deriv = b.I * inc + (m.T * h.T * r.I) * (h * m * inc - d)

  return j_deriv.A.flatten() * 2.0

