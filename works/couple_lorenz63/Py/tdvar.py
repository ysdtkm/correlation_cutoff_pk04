#!/usr/bin/env python

import numpy as np
import sys
from scipy.optimize import fmin_bfgs
from const import *
import stats_const
import better_exceptions

def tdvar(fcst, h, r, yo, i_s, i_e, amp_b):
  # fcst   <- np.array[dimc]        : first guess
  # h      <- np.array[DIMO, dimc]  : observation operator
  # r      <- np.array[DIMO, DIMO]  : observation error covariance
  # yo     <- np.array[DIMO, 1]     : observation
  # i_s    <- int
  # i_e    <- int
  # amp_b  <- float
  # return -> np.array[dimc]        : assimilated field
  anl = np.copy(fcst)
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo, i_s, i_e, amp_b), disp=False)
  return anl.flatten()

def tdvar_2j(anl_nda, fcst_nda, h_nda, r_nda, yo_nda, i_s, i_e, amp_b):
  # anl_nda  <- np.array[dimc]        : temporary analysis field
  # fcst_nda <- np.array[dimc]        : first guess field
  # h_nda    <- np.array[DIMO, dimc]  : observation operator
  # r_nda    <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda   <- np.array[DIMO, 1]     : observation
  # i_s      <- int                   : first model grid to assimilate
  # i_e      <- int                   : last model grid to assimilate
  # amp_b    <- float
  # return   -> float                 : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])

  anl  = np.asmatrix(anl_nda).T
  fcst = np.asmatrix(fcst_nda).T
  twoj = (anl - fcst).T * b.I * (anl - fcst) + \
       (h * anl - yo).T * r.I * (h * anl - yo)
  return twoj[0,0]

def tdvar_interpol(fcst, h_nda, r_nda, yo_nda, i_s, i_e, amp_b):
  # Return same analysis with tdvar(), not by minimization but analytical interpolation
  # fcst   <- np.array[dimc]        : first guess
  # h_nda  <- np.array[DIMO, dimc]  : observation operator
  # r_nda  <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda <- np.array[DIMO, 1]     : observation
  # i_s    <- int
  # i_e    <- int
  # amp_b  <- float
  # return -> np.array[dimc]        : assimilated field

  xb = np.asmatrix(fcst).T
  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])

  d = yo - h * xb

  inc_model = (b.I + h.T * r.I * h).I * h.T * r.I * d
  anl = (xb + inc_model).A.flatten()
  return anl

