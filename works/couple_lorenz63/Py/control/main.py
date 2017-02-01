#!/usr/bin/env python

import numpy as np
import sys, os
py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_dir)
from control.const import *
from model.model import *
from assim.etkf import *
from assim.tdvar import *
from assim.fdvar import *

def main():
  nature = exec_nature()
  for exp in EXPLIST:
    obs  = exec_obs(exp, nature)
    free = exec_free_run(exp)
    anl  = exec_assim_cycle(exp, free, obs)
    # sys.exit()
    # exec_fcst(fcst)

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
  # np.random.seed(1)
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
  return all_fcst

def exec_fcst():
  all_fcst.tofile("data/%s_fcst.bin" % exp[0])
  return 0

main()

