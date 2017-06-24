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
    # and then amplitude modified so that trace(B) = DIMM
    b = np.array([ \
      [  1.12248884,   1.57187271, 0.00384028748, 0.00305089239, 0.0132861318, -0.00527071038, 0.000129830546, 0.0012284577, 0.00168635187], \
      [  1.57187271,   2.78969539, 0.0350119689, 0.00613827406, 0.0217481499, -0.00754758442, 0.000366555651, 0.00298389039, 0.00187924681], \
      [0.00384028748, 0.0350119689,   2.16425386, 0.000773735041, 0.00128326765, 0.000125953175, 0.000336935559, -0.000301960444, 0.0028894971], \
      [0.00305089239, 0.00613827406, 0.000773735041,  0.185766489,  0.217713887, -0.0681886567, -0.00731521182, 0.0725531368, 0.0238609046], \
      [0.0132861318, 0.0217481499, 0.00128326765,  0.217713887,  0.389668892, 0.00778414136, 0.0130640687,  0.107573181, 0.0249755739], \
      [-0.00527071038, -0.00754758442, 0.000125953175, -0.0681886567, 0.00778414136,   0.42421797, -0.00795050824, -0.0536740054, 0.00129324384], \
      [0.000129830546, 0.000366555651, 0.000336935559, -0.00731521182, 0.0130640687, -0.00795050824,   0.13751132,  0.270033759,  0.108649225], \
      [0.0012284577, 0.00298389039, -0.000301960444, 0.0725531368,  0.107573181, -0.0536740054,  0.270033759,   1.02331517, 0.0629941266], \
      [0.00168635187, 0.00187924681, 0.0028894971, 0.0238609046, 0.0249755739, 0.00129324384,  0.108649225, 0.0629941266,  0.763082077]  \
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

