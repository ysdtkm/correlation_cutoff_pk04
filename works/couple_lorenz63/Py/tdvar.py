#!/usr/bin/env python

import numpy as np
from scipy.optimize import fmin_bfgs
from const import *

def tdvar(fcst, h, r, yo, i_s, i_e):
  ### here, (dimm = i_e - i_s <= DIMM) unless strongly coupled
  # fcst   <- np.array[dimm]        : first guess
  # h      <- np.array[DIMO, dimm]  : observation operator
  # r      <- np.array[DIMO, DIMO]  : observation error covariance
  # yo     <- np.array[DIMO, 1]     : observation
  # i_s    <- int
  # i_e    <- int
  # return -> np.array[dimm]        : assimilated field
  anl = np.copy(fcst)
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo, i_s, i_e), disp=False)
  return anl.T

def tdvar_2j(anl_nda, fcst_nda, h_nda, r_nda, yo_nda, i_s, i_e):
  ### here, (dimm = i_e - i_s <= DIMM) unless strongly coupled
  # anl_nda  <- np.array[dimm]        : temporary analysis field
  # fcst_nda <- np.array[dimm]        : first guess field
  # h_nda    <- np.array[DIMO, dimm]  : observation operator
  # r_nda    <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda   <- np.array[DIMO, 1]     : observation
  # i_s      <- int
  # i_e      <- int
  # return   -> float                 : cost function 2J

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

  # todo: use realistic B
  b = np.diag(np.ones(DIMM))
  return b

