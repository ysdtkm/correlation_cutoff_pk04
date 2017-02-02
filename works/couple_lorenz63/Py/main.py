#!/usr/bin/env python

import numpy as np
from const import *
from model import *
from etkf import *
from tdvar import *
from fdvar import *

def main():
  np.random.seed(1)
  nature = exec_nature()
  for exp in EXPLIST:
    obs  = exec_obs(exp, nature)
    free = exec_free_run(exp)
    anl  = exec_assim_cycle(exp, free, obs)
    exec_fcst(exp, anl)

def exec_nature():
  all_true = np.empty((STEPS, DIMM))
  true = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(0, STEPS):
    true[:] = timestep(true[:])
    all_true[i,:] = true[:]
  all_true.tofile("data/true.bin")
  return all_true

def exec_obs(exp, nature):
  all_obs = np.empty((STEPS, DIMO))
  all_obs[:,:] = np.nan
  for i in range(0, STEPS):
    if (i % exp["aint"] == 0):
      all_obs[i,:] = nature[i] + np.random.normal(0.0, OERR, DIMO)
  all_obs.tofile("data/%s_obs.bin" % exp["name"])
  return all_obs

def exec_free_run(exp):
  free_run = np.empty((STEPS, exp["nmem"], DIMM))
  for m in range(0, exp["nmem"]):
    free_run[0,m,:] = np.random.normal(0.0, FERR_INI, DIMM)
    for i in range(1, STEP_FREE):
      free_run[i,m,:] = timestep(free_run[i-1,m,:])
  return free_run

def exec_assim_cycle(exp, all_fcst, all_obs):
  # forecast-analysis cycle
  r = getr()
  h = geth(exp["diag"])
  fcst = np.empty((exp["nmem"], DIMM))
  for i in range(STEP_FREE, STEPS):
    for m in range(0, exp["nmem"]):
      fcst[m,:] = timestep(all_fcst[i-1,m,:])
    if (i % exp["aint"] == 0):
      yo = h * np.matrix(all_obs[i]).T
      if (exp["method"] == "etkf"):
        fcst[:,:] = etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], exp["inf"], exp["nmem"], i)
      elif (exp["method"] == "3dvar"):
        fcst[0,:] = tdvar(np.matrix(fcst[:,:]).T, h[:,:], r[:,:], yo[:,:])
      elif (exp["method"] == "4dvar"):
        fcst[0,:] = fdvar(np.matrix(all_fcst[i-exp["aint"],:,:]).T, \
          h[:,:], r[:,:], yo[:,:], exp["aint"])
    all_fcst[i,:,:] = fcst[:,:]
  all_fcst.tofile("data/%s_cycle.bin" % exp["name"])
  return all_fcst

def exec_fcst(exp, anl):
  fcst_all = np.empty((STEPS, FCST_LT, exp["nmem"], DIMM))
  for i in range(STEP_FREE, STEPS):
    if (i % exp["aint"] == 0):
      for m in range(0, exp["nmem"]):
        for lt in range(1, FCST_LT):
          fcst_all[i,lt,m,:] = timestep(fcst_all[i-1,lt,m,:])
  fcst_all.tofile("data/%s_fcst.bin" % exp["name"])
  return 0

main()

