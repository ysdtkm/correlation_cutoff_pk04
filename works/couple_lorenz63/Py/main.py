#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin_bfgs
from const import *

def main():
  for exp in EXPLIST:
    aint = exp[2]
    nmem = exp[4]
    method = exp[5]
    all_true = np.empty((STEPS, DIMM))
    all_fcst = np.empty((STEPS, nmem, DIMM))
    all_obs  = np.empty((STEPS, DIMO))
    h = geth(exp[3])
    r = getr()
    np.random.seed(1)

    # initialization with free run
    true = np.random.normal(0.0, FERR_INI, DIMM)
    fcst = np.empty((nmem, DIMM))
    for m in range(0, nmem):
      fcst[m,:] = np.random.normal(0.0, FERR_INI, DIMM)
    for i in range(0, int(50 / DT)):
      true = timestep(true)
      for m in range(0, nmem):
        fcst[m,:] = timestep(fcst[m,:])

    # forecast-analysis cycle
    for i in range(0, STEPS):
      true[:] = timestep(true[:])
      all_true[i,:] = true[:]
      for m in range(0, nmem):
        fcst[m,:] = timestep(fcst[m,:])
      if (i % aint == 0) & (i != 0):
        yo = h * np.matrix(true).T + np.matrix(np.random.normal(0.0, OERR, DIMO)).T
        all_obs[i,:] = yo.A.flatten()
        if (method == "etkf"):
          fcst[:,:] = etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], exp[1], nmem, i)
        elif (method == "3dvar"):
          fcst[0,:] = tdvar(np.matrix(fcst[:,:]).T, h[:,:], r[:,:], yo[:,:])
        elif (method == "4dvar"):
          fcst[0,:] = fdvar(np.matrix(all_fcst[i-aint,:,:]).T, \
            h[:,:], r[:,:], yo[:,:], aint)
      else:
        all_obs[i,:] = np.nan
      all_fcst[i,:,:] = fcst[:,:]

    all_true.tofile("data/%s_true.bin" % exp[0])
    all_fcst.tofile("data/%s_fcst.bin" % exp[0])
    all_obs.tofile("data/%s_obs.bin" % exp[0])

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

def tdvar(fcst, h, r, yo):
  anl = fcst
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo))
  return anl.T

def fdvar(fcst_0, h, r, yo, aint):
  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time
  anl_0 = fcst_0
  anl_0 = fmin_bfgs(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint))
  anl_1 = anl_0
  for i in range(0, aint):
    anl_1 = timestep(anl_1)
  return anl_1.T

def tdvar_2j(anl, fcst, h, r, yo):
  b = 1.2 * tdvar_b()
  anl_tmp = anl
  anl = np.matrix(anl_tmp).T
  twoj = (anl - fcst).T * np.linalg.inv(b) * (anl - fcst) + \
       (h * anl - yo).T * np.linalg.inv(r) * (h * anl - yo)
  return twoj

def fdvar_2j(anl_0, fcst_0, h, r, yo, aint):
  b = 0.6 * 1.2 * tdvar_b()
  anl_tmp = anl_0
  anl_0 = np.matrix(anl_tmp).T
  anl_1_ar = anl_0.A.flatten()
  for i in range(0, aint):
    anl_1_ar = timestep(anl_1_ar)
  anl_1 = np.matrix(anl_1_ar).T
  twoj = (anl_0 - fcst_0).T * np.linalg.inv(b) * (anl_0 - fcst_0) + \
       (h * anl_1 - yo).T * np.linalg.inv(r) * (h * anl_1 - yo)
  return twoj.A

def tdvar_b():
  # obtained from off-line ETKF (500-1999th step mean(xbpt*xbpt.T))
  b = np.matrix([ \
    [ 0.34897187,0.5587116,-0.08293288], \
    [ 0.5587116, 1.10975806,0.00229167], \
    [-0.08293288,0.00229167,0.60078791]  \
  ])
  return b

def getr():
  r = np.matrix(np.identity(DIMO)) * (OERR * OERR)
  return r

def geth(diag_h):
  # DIMO == DIMM is assumed
  h = np.matrix(np.zeros((DIMO,DIMM)))
  for i in range(0, DIMM):
    h[i,i] = diag_h[i]
  return h

def timestep(x):
  # np.array x[DIMM]
  k1 = tendency(x)
  x2 = x + k1 * DT / 2.0
  k2 = tendency(x2)
  x3 = x + k2 * DT / 2.0
  k3 = tendency(x3)
  x4 = x + k3 * DT
  k4 = tendency(x4)
  x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * DT / 6.0
  return x

def tendency(x):
  sigma = 10.0
  r = 28.0
  b = 8.0 / 3.0

  k = np.empty((DIMM))
  k[0] = -sigma * x[0]            + sigma * x[1]
  k[1] =  -x[0] * x[2] + r * x[0] -         x[1]
  k[2] =   x[0] * x[1] - b * x[2]
  return k

main()

