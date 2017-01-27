#!/usr/bin/env python

import numpy as np
import sys, os
sys.path.append(os.pardir)
from const import *
from model.model import *

def main():
  exec_nature()
  sys.exit()
  exec_obs()
  exec_free_run()
  for exp in EXPLIST:
    exec_assim(exp)
    exec_fcst(fcst)

def exec_nature():
  all_true = np.empty((STEPS, DIMM))
  true = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(0, STEPS):
    true[:] = timestep(true[:])
    all_true[i,:] = true[:]
  all_true.tofile("data/true.bin")
  return 0

def exec_obs():
  all_obs  = np.empty((STEPS, DIMO))
  all_obs.tofile("data/%s_obs.bin" % exp[0])
  return 0

def exec_free_run():
  all_fcst = np.empty((STEPS, nmem, DIMM))
  h = geth(exp[3])
  r = getr()
  np.random.seed(1)

  # initialization with free run
  fcst = np.empty((exp["nmem"], DIMM))
  for m in range(0, exp["nmem"]):
    fcst[m,:] = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(0, int(50 / DT)):
    true = timestep(true)
    for m in range(0, exp["nmem"]):
      fcst[m,:] = timestep(fcst[m,:])
  return 0

def exec_assim(exp):
  # forecast-analysis cycle
  for i in range(0, STEPS):
    for m in range(0, exp["nmem"]):
      fcst[m,:] = timestep(fcst[m,:])
    if (i % exp["aint"] == 0) & (i != 0):
      yo = h * np.matrix(true).T + np.matrix(np.random.normal(0.0, OERR, DIMO)).T
      all_obs[i,:] = yo.A.flatten()
      if (exp["method"] == "etkf"):
        fcst[:,:] = etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], exp[1], exp["nmem"], i)
      elif (exp["method"] == "3dvar"):
        fcst[0,:] = tdvar(np.matrix(fcst[:,:]).T, h[:,:], r[:,:], yo[:,:])
      elif (exp["method"] == "4dvar"):
        fcst[0,:] = fdvar(np.matrix(all_fcst[i-exp["aint"],:,:]).T, \
          h[:,:], r[:,:], yo[:,:], exp["aint"])
    else:
      all_obs[i,:] = np.nan
    all_fcst[i,:,:] = fcst[:,:]

def exec_fcst():
  all_fcst.tofile("data/%s_fcst.bin" % exp[0])
  return 0

main()

