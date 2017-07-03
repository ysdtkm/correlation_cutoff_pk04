#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import *

def etkf(fcst, h_nda, r_nda, yo_nda, rho, nmem, localization=False):
  # fcst   <- np.array[nmem, dimc]
  # h_nda  <- np.array[DIMO, dimc]
  # r_nda  <- np.array[DIMO, DIMO]
  # yo_nda <- np.array[DIMO, 1]
  # rho    <- float
  # nmem   <- int
  # return -> np.array[dimc, nmem], np.array[dimc, dimc], np.array[dimc, dimc]

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)

  ### DA variables (np.matrix)
  # - model space
  #  xfm[dimc,nmem] : each member's forecast Xf
  # xfpt[dimc,nmem] : forecast perturbation Xbp
  #  xam[dimc,nmem] : each member's analysis Xa
  # xapt[dimc,nmem] : analysis perturbation Xap
  #   xf[dimc,1   ] : ensemble mean forecast xf_bar
  #   xa[dimc,1   ] : ensemble mean analysis xa_bar
  #
  # - obs space
  #    r[DIMO,DIMO] : obs error covariance matrix R
  #    h[DIMO,dimc] : linearized observation operator H
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

  if localization:
    xai = np.matrix(np.zeros((DIMM, nmem)))
    xapt = np.copy(xfpt)

    for j in range(DIMM):
      # step 3: need elaboration for real localization
      localization_weight = np.matrix(np.identity(DIMM))
      for ix in range(DIMM):
        if ix != j:
          localization_weight[ix, ix] *= 0.5
      yol = yo[:,:]
      ybl = yb[:,:]
      ybptl = ybpt[:,:]
      xfl = xf[j,:]
      xfptl = xfpt[j,:]
      rl = r[:,:]

      # step 4-9
      cl = ybptl.T * np.asmatrix(rl.I.A * localization_weight.A)
      pal = (((nmem-1.0)/rho) * I_mm + cl * ybptl).I
      waptl = np.matrix(sqrtm((nmem-1.0) * pal))
      wal = pal * cl * (yol - ybl)
      xail = xfl * I_1m + xfptl * (wal * I_1m + waptl)
      xai[j,:] = xail[:,:]
    return np.real(xai.T.A), (xfpt * xfpt.T).A, None

  else:
    pa   = (((nmem-1.0)/rho) * I_mm + ybpt.T * r.I * ybpt).I
    wam  = np.matrix(sqrtm((nmem-1.0) * pa))
    wa   = pa * ybpt.T * r.I * (yo - yb)
    xapt = (xfm - xf * I_1m) * wam
    xa   = xf + xfm * wa
    xam  = xapt + xa * I_1m
    return np.real(xam.T.A), (xfpt * xfpt.T).A, (xapt * xapt.T).A

