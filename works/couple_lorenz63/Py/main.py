#!/usr/bin/env python

import numpy as np
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

main()

