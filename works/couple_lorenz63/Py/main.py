#!/usr/bin/env python

import numpy as np
from const import *
from model import *
from etkf import *
from tdvar import *
from fdvar import *
from vectors import *

def main():
  np.random.seed(1)
  nature = exec_nature()
  obs = exec_obs(nature)
  for exp in EXPLIST:
    free = exec_free_run(exp)
    anl  = exec_assim_cycle(exp, free, obs)
    exec_deterministic_fcst(exp, anl)

def exec_nature():
  # return   -> np.array[STEPS, DIMM]
  # file1    -> np.array[STEPS, DIMM]       : nature
  # file2    -> np.array[STEPS, DIMM, DIMM] : backward LVs
  # file3    -> np.array[STEPS, DIMM]       : LEs

  all_true = np.empty((STEPS, DIMM))
  true = np.random.normal(0.0, FERR_INI, DIMM)

  eps = 1.0e-9
  all_lv = np.empty((STEPS, DIMM, DIMM))
  lv = np.random.normal(0.0, eps, (DIMM, DIMM))
  lv, le = orth_norm_vectors(lv, eps)
  all_le = np.zeros((STEPS, DIMM))

  for i in range(0, STEPS):
    true[:] = timestep(true[:], DT)
    all_true[i,:] = true[:]

    m = finite_time_tangent_using_nonlinear(true, DT, 1)
    lv = m * lv
    if (i % 1000 == 0):
      lv, le = orth_norm_vectors(lv, eps)
      all_le[i,:] = le[:]
    all_lv[i,:,:] = lv[:,:]

  all_true.tofile("data/true.bin")
  all_lv.tofile("data/lv.bin")
  all_le.tofile("data/le.bin")
  print(np.mean(all_le[STEPS//2:,:], axis=0))
  return all_true

def exec_obs(nature):
  # nature   <- np.array[STEPS, DIMM]
  # return   -> np.array[STEPS, DIMO]

  all_obs = np.empty((STEPS, DIMO))
  for i in range(0, STEPS):
    all_obs[i,:] = nature[i] + np.random.normal(0.0, OERR, DIMO)
  all_obs.tofile("data/obs.bin")
  return all_obs

def exec_free_run(exp):
  # exp      <- hash
  # return   -> np.array[STEPS, nmem, DIMM]

  free_run = np.empty((STEPS, exp["nmem"], DIMM))
  for m in range(0, exp["nmem"]):
    free_run[0,m,:] = np.random.normal(0.0, FERR_INI, DIMM)
    for i in range(1, STEP_FREE):
      free_run[i,m,:] = timestep(free_run[i-1,m,:], DT)
  return free_run

def exec_assim_cycle(exp, all_fcst, all_obs):
  # exp      <- hash
  # all_fcst <- np.array[STEPS, nmem, DIMM]
  # all_obs  <- np.array[STEPS, DIMO]
  # return   -> np.array[STEPS, nmem, DIMM]

  # forecast-analysis cycle
  r = getr()
  h = geth(exp["diag"])
  fcst = np.empty((exp["nmem"], DIMM))
  all_ba = np.empty((STEPS, DIMM, DIMM))
  all_ba[:,:,:] = np.nan
  all_bf = np.empty((STEPS, DIMM, DIMM))
  all_bf[:,:,:] = np.nan
  obs_used = np.empty((STEPS, DIMO))
  obs_used[:,:] = np.nan
  for i in range(STEP_FREE, STEPS):
    for m in range(0, exp["nmem"]):
      fcst[m,:] = timestep(all_fcst[i-1,m,:], DT)
    if (i % exp["aint"] == 0):
      obs_used[i,:] = all_obs[i,:]
      yo = h * np.matrix(all_obs[i,:]).T
      if (exp["method"] == "etkf"):
        fcst[:,:], all_bf[i,:,:], all_ba[i,:,:] = \
          etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], exp["inf"], exp["nmem"])
      elif (exp["method"] == "3dvar"):
        fcst[0,:] = tdvar(np.matrix(fcst[:,:]).T, h[:,:], r[:,:], yo[:,:])
      elif (exp["method"] == "4dvar"):
        fcst[0,:] = fdvar(np.matrix(all_fcst[i-exp["aint"],:,:]).T, \
          h[:,:], r[:,:], yo[:,:], exp["aint"])
    all_fcst[i,:,:] = fcst[:,:]
  obs_used.tofile("data/%s_obs.bin" % exp["name"])
  all_fcst.tofile("data/%s_cycle.bin" % exp["name"])
  all_bf.tofile("data/%s_covr_back.bin" % exp["name"])
  all_ba.tofile("data/%s_covr_anl.bin" % exp["name"])
  return all_fcst

def exec_deterministic_fcst(exp, anl):
  # exp    <- hash
  # anl    <- np.array[STEPS, nmem, DIMM]
  # return -> np.array[STEPS, FCST_LT, DIMM]

  fcst_all = np.empty((STEPS, FCST_LT, DIMM))
  for i in range(STEP_FREE, STEPS):
    if (i % exp["aint"] == 0):
      fcst_all[i,0,:] = np.mean(anl[i,:,:], axis=0)
      for lt in range(1, FCST_LT):
        fcst_all[i,lt,:] = timestep(fcst_all[i-1,lt,:], DT)
  fcst_all.tofile("data/%s_fcst.bin" % exp["name"])
  return 0

main()

