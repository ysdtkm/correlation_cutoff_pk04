#!/usr/bin/env python

import numpy as np
from const import *

def etkf(fcst, h, r, yo, inf, nmem, istep):
  ### DA variables
  # - model space
  #  xfm[DIMM,nmem] : each member's forecast Xf
  # xfpt[DIMM,nmem] : forecast perturbation Xbp
  #  xam[DIMM,nmem] : each member's analysis Xa
  # xapt[DIMM,nmem] : analysis perturbation Xap
  #   xf[DIMM,1   ] : ensemble mean forecast xf_bar
  #   xa[DIMM,1   ] : ensemble mean analysis xa_bar
  #
  # - obs space
  #    r[DIMO,DIMO] : obs error covariance matrix R
  #    h[DIMO,DIMM] : linearized observation operator H
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
  pa   = np.linalg.inv(((nmem-1.0)/inf**2) * I_mm \
                      + ybpt.T * np.linalg.inv(r) * ybpt)
  wam  = np.matrix(sqrtm((nmem-1.0) * pa))
  wa   = pa * ybpt.T * np.linalg.inv(r) * (yo - yb)
  xapt = (xfm - xf * I_1m) * wam
  xa   = xf + xfm * wa
  xam  = xapt + xa * I_1m
  return np.real(xam.T.A)

