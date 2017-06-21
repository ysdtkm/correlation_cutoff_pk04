#!/usr/bin/env python

import numpy as np
import sys
from scipy.optimize import fmin_bfgs
from const import *

Amplitude_B = 0.6

def tdvar(fcst, h, r, yo, i_s, i_e):
  # fcst   <- np.array[dimc]        : first guess
  # h      <- np.array[DIMO, dimc]  : observation operator
  # r      <- np.array[DIMO, DIMO]  : observation error covariance
  # yo     <- np.array[DIMO, 1]     : observation
  # i_s    <- int
  # i_e    <- int
  # return -> np.array[dimc]        : assimilated field
  anl = np.copy(fcst)
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo, i_s, i_e), disp=False)
  return anl.flatten()

def tdvar_2j(anl_nda, fcst_nda, h_nda, r_nda, yo_nda, i_s, i_e):
  # anl_nda  <- np.array[dimc]        : temporary analysis field
  # fcst_nda <- np.array[dimc]        : first guess field
  # h_nda    <- np.array[DIMO, dimc]  : observation operator
  # r_nda    <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda   <- np.array[DIMO, 1]     : observation
  # i_s      <- int                   : first model grid to assimilate
  # i_e      <- int                   : last model grid to assimilate
  # return   -> float                 : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(Amplitude_B * tdvar_b()[i_s:i_e, i_s:i_e])

  anl  = np.asmatrix(anl_nda).T
  fcst = np.asmatrix(fcst_nda).T
  twoj = (anl - fcst).T * b.I * (anl - fcst) + \
       (h * anl - yo).T * r.I * (h * anl - yo)
  return twoj[0,0]

def tdvar_interpol(fcst, h_nda, r_nda, yo_nda, i_s, i_e):
  # Return same analysis with tdvar(), not by minimization but analytical interpolation
  # fcst   <- np.array[dimc]        : first guess
  # h_nda  <- np.array[DIMO, dimc]  : observation operator
  # r_nda  <- np.array[DIMO, DIMO]  : observation error covariance
  # yo_nda <- np.array[DIMO, 1]     : observation
  # i_s    <- int
  # i_e    <- int
  # return -> np.array[dimc]        : assimilated field

  xb = np.asmatrix(fcst).T
  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(Amplitude_B * tdvar_b()[i_s:i_e, i_s:i_e])

  d = yo - h * xb

  inc_model = (b.I + h.T * r.I * h).I * h.T * r.I * d
  anl = (xb + inc_model).A.flatten()
  return anl

def tdvar_b():
  # return -> np.array[DIMM,DIMM] : background error covariance

  if DIMM == 9:
    b = np.array( \
      [[1.55369946e-02, 2.59672759e-02, 2.14143702e-02, 5.50608636e-05, 2.06837903e-04, 9.53104122e-05, 4.01854229e-05, 1.10394688e-04, 1.10795043e-04], \
       [2.59672759e-02, 4.60323124e-02, 3.22253391e-02, 9.67994015e-05, 3.47479991e-04, 1.59885557e-04, 6.25825607e-05, 1.71056662e-04, 1.75287574e-04], \
       [2.14143702e-02, 3.22253391e-02, 4.49870690e-02, 8.07242543e-05, 2.90997346e-04, 1.68933576e-04, 7.17501951e-05, 1.91677973e-04, 2.08898907e-04], \
       [5.50608636e-05, 9.67994015e-05, 8.07242543e-05, 1.19388552e-04, 1.32465565e-04, 1.41518295e-04, 1.26069316e-04, 3.89377305e-04, 5.26723750e-04], \
       [2.06837903e-04, 3.47479991e-04, 2.90997346e-04, 1.32465565e-04, 2.19079467e-04, 1.28264891e-04, 2.15409763e-04, 5.82014129e-04, 7.50531420e-04], \
       [9.53104122e-05, 1.59885557e-04, 1.68933576e-04, 1.41518295e-04, 1.28264891e-04, 3.00276052e-04, 2.90918095e-04, 8.40635972e-04, 8.63515366e-04], \
       [4.01854229e-05, 6.25825607e-05, 7.17501951e-05, 1.26069316e-04, 2.15409763e-04, 2.90918095e-04, 9.88628471e-04, 2.56406108e-03, 2.73643907e-03], \
       [1.10394688e-04, 1.71056662e-04, 1.91677973e-04, 3.89377305e-04, 5.82014129e-04, 8.40635972e-04, 2.56406108e-03, 8.41665768e-03, 6.19483319e-03], \
       [1.10795043e-04, 1.75287574e-04, 2.08898907e-04, 5.26723750e-04, 7.50531420e-04, 8.63515366e-04, 2.73643907e-03, 6.19483319e-03, 1.08456084e-02]] \
    ) * 100.0 * 10.0

  elif DIMM == 3:
    b = np.array([ \
      [ 0.34897187,0.5587116,-0.08293288], \
      [ 0.5587116, 1.10975806,0.00229167], \
      [-0.08293288,0.00229167,0.60078791] \
    ])

  else:
    b = np.diag(np.ones(DIMM)) * 1.5

  return b

