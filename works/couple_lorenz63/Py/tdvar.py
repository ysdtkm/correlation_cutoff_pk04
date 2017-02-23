#!/usr/bin/env python

import numpy as np
from scipy.optimize import fmin_bfgs
from const import *

def tdvar(fcst, h, r, yo, i_s, i_e):
  # fcst   <- np.array[DIMM]        : first guess
  # h      <- np.array[DIMO, DIMM]  : observation operator
  # r      <- np.array[DIMO, DIMO]  : observation error covariance
  # yo     <- np.array[DIMO, 1]     : observation
  # return -> np.array[DIMM]        : assimilated field
  anl = np.copy(fcst)
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo, i_s, i_e))
  return anl.T

def tdvar_2j(anl_nda, fcst_nda, h_nda, r_nda, yo_nda, i_s, i_e):
  # anl_nda   <- np.array[DIMM]        : temporary analysis field
  # fcst_nda  <- np.array[DIMM]        : first guess field
  # h_nda     <- np.array[DIMO, DIMM]  : observation operator
  # r_nda     <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda    <- np.array[DIMO, 1]     : observation
  # return    -> float                 : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(1.2 * tdvar_b()[i_s:i_e, i_s:i_e])

  anl  = np.asmatrix(anl_nda).T
  fcst = np.asmatrix(fcst_nda).T
  twoj = (anl - fcst).T * b.I * (anl - fcst) + \
       (h * anl - yo).T * r.I * (h * anl - yo)
  return twoj

def tdvar_b():
  # return -> np.array[DIMM,DIMM] : background error covariance

  # obtained from off-line ETKF (500-1999th step mean(xbpt*xbpt.T))
  # b = np.matrix([ \
  #   [ 0.34897187,0.5587116,-0.08293288], \
  #   [ 0.5587116, 1.10975806,0.00229167], \
  #   [-0.08293288,0.00229167,0.60078791]  \
  # ])
  b = np.diag(np.ones(DIMM))
  return b

