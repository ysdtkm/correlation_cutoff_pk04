#!/usr/bin/env python

import numpy as np
from scipy.optimize import fmin_bfgs
from const import *

def tdvar(fcst, h, r, yo):
  # fcst   <- np.array[DIMM]        : first guess
  # h      <- np.array[DIMO, DIMM] : observation operator
  # r      <- np.array[DIMO, DIMO] : observation error covariance
  # yo     <- np.array[DIMO, 1]    : observation
  # return -> np.array[DIMM]        : assimilated field
  anl = fcst
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo))
  return anl.T

def tdvar_2j(anl, fcst, h_nda, r_nda, yo_nda):
  # anl    <- np.array[DIMM]        : temporary analysis field
  # fcst   <- np.array[DIMM]        : first guess field
  # h_nda  <- np.array[DIMO, DIMM]  : observation operator
  # r_nda  <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda <- np.array[DIMO, 1]     : observation
  # return -> float                 : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(1.2 * tdvar_b())

  anl_tmp = anl
  anl = np.matrix(anl_tmp).T
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

