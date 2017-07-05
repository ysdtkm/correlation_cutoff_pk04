#!/usr/bin/env python

import numpy as np
import sys
from scipy.optimize import fmin_bfgs
from const import *

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
  b  = np.matrix(amp_b * tdvar_b()[i_s:i_e, i_s:i_e])

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
  b  = np.matrix(amp_b * tdvar_b()[i_s:i_e, i_s:i_e])

  d = yo - h * xb

  inc_model = (b.I + h.T * r.I * h).I * h.T * r.I * d
  anl = (xb + inc_model).A.flatten()
  return anl

def tdvar_b():
  # return -> np.array[DIMM,DIMM] : background error covariance

  if DIMM == 9:
    # obtained by unit_test.py/obtain_tdvar_b(), a66aa31, 100000 timesteps
    b = np.array([
    [ 0.151366652,  0.217993747, 0.00074922173, 0.000654576057, 0.00266310433, -0.00218875723, 0.000794892471, 0.00598052358, -0.00862711731],
    [ 0.217993747,  0.407210895, 0.00539653061, 0.0013996398, 0.00445064454, -0.00249628423, 0.00109589198, 0.00926521126, -0.0124801346],
    [0.00074922173, 0.00539653061,  0.351884549, -0.000379190086, -0.000347220003, 0.000498117183, -0.000366326131, 0.00148373321, -0.00485325959],
    [0.000654576057, 0.0013996398, -0.000379190086, 0.0776620651, 0.0939750707, -0.0179157645, 0.00932743722,  0.109199709, 0.0786195063],
    [0.00266310433, 0.00445064454, -0.000347220003, 0.0939750707,   0.16266338, -0.00415181247, 0.0706870452,  0.255559538,  0.139336958],
    [-0.00218875723, -0.00249628423, 0.000498117183, -0.0179157645, -0.00415181247,  0.188983318, -0.0589431514, -0.173905592, 0.0943891795],
    [0.000794892471, 0.00109589198, -0.000366326131, 0.00932743722, 0.0706870452, -0.0589431514,  0.561413675,   1.10104363,  0.465250176],
    [0.00598052358, 0.00926521126, 0.00148373321,  0.109199709,  0.255559538, -0.173905592,   1.10104363,   4.04932551,  0.227404405],
    [-0.00862711731, -0.0124801346, -0.00485325959, 0.0786195063,  0.139336958, 0.0943891795,  0.465250176,  0.227404405,   3.04948996] 
    ])




  elif DIMM == 3:
    b = np.array([ \
      [ 0.34897187,0.5587116,-0.08293288], \
      [ 0.5587116, 1.10975806,0.00229167], \
      [-0.08293288,0.00229167,0.60078791] \
    ])

  else:
    b = np.diag(np.ones(DIMM)) * 1.5

  return b

