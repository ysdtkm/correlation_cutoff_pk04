#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import *

def etkf(fcst, h_nda, r_nda, yo_nda, inf, nmem):
  ### here, (dimm = i_e - i_s <= DIMM) unless strongly coupled
  # fcst   <- np.array[nmem, dimm]
  # h_nda  <- np.array[DIMO, dimm]
  # r_nda  <- np.array[DIMO, DIMO]
  # yo_nda <- np.array[DIMO, 1]
  # inf    <- float
  # nmem   <- int
  # return -> np.array[dimm, nmem], np.array[dimm, dimm], np.array[dimm, dimm]

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)

  ### DA variables (np.matrix)
  # - model space
  #  xfm[dimm,nmem] : each member's forecast Xf
  # xfpt[dimm,nmem] : forecast perturbation Xbp
  #  xam[dimm,nmem] : each member's analysis Xa
  # xapt[dimm,nmem] : analysis perturbation Xap
  #   xf[dimm,1   ] : ensemble mean forecast xf_bar
  #   xa[dimm,1   ] : ensemble mean analysis xa_bar
  #
  # - obs space
  #    r[DIMO,DIMO] : obs error covariance matrix R
  #    h[DIMO,dimm] : linearized observation operator H
  #   yo[DIMO,1   ] : observed state yo
  #   yb[DIMO,1   ] : mean simulated observation vector yb
  #  ybm[DIMO,nmem] : ensemble model simulated observation matrix Yb
  #
  # - mem space
  #  wam[nmem,nmem] : optimal weight matrix for each member
  #   wa[nmem,1   ] : optimal weight for each member
  #   pa[nmem,nmem] : approx. anl error covariance matrix Pa in ens space
  #
  # note :
  #   DIMM > DIMO >> nmem in typical LETKF application
  ###
  I_mm = np.matrix(np.identity(nmem))
  I_1m = np.matrix(np.ones((1,nmem)))

  xfm  = np.matrix(fcst[:,:]).T
  xf   = np.mean(xfm, axis=1)
  xfpt = xfm - xf * I_1m
  ybpt = h * xfpt
  yb   = h * xf
  pa   = (((nmem-1.0)/inf**2) * I_mm + ybpt.T * r.I * ybpt).I
  wam  = np.matrix(sqrtm((nmem-1.0) * pa))
  wa   = pa * ybpt.T * r.I * (yo - yb)
  xapt = (xfm - xf * I_1m) * wam
  xa   = xf + xfm * wa
  xam  = xapt + xa * I_1m
  return np.real(xam.T.A), (xfpt * xfpt.T).A, (xapt * xapt.T).A

