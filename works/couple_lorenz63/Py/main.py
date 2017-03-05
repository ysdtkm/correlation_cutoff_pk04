#!/usr/bin/env python

import os
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
  # file3    -> np.array[STEPS, DIMM]       : BLEs
  # file4    -> np.array[STEPS, DIMM, DIMM] : forward LVs
  # file5    -> np.array[STEPS, DIMM]       : FLEs

  all_true = np.empty((STEPS, DIMM))
  true = np.random.normal(0.0, FERR_INI, DIMM)
  eps = 1.0e-9
  orth_int = 1

  # forward integration i-1 -> i
  all_blv = np.empty((STEPS, DIMM, DIMM))
  blv = np.random.normal(0.0, eps, (DIMM, DIMM))
  blv, ble = orth_norm_vectors(blv, eps)
  all_ble = np.zeros((STEPS, DIMM))
  for i in range(0, STEPS):
    m = finite_time_tangent_using_nonlinear(true, DT, 1)
    true[:] = timestep(true[:], DT)
    all_true[i,:] = true[:]
    blv = np.dot(m, blv)
    if (i % orth_int == 0):
      # (ble[:] / DT) is (orth_int * actual LEs).
      # For window without orthonormalization, LEs are zero.
      # Thus long-term mean wil be actual LEs.
      blv, ble = orth_norm_vectors(blv, eps)
      all_ble[i,:] = ble[:] / DT
    all_blv[i,:,:] = blv[:,:]

  # backward integration i-1 <- i
  all_flv = np.empty((STEPS, DIMM, DIMM))
  flv = np.random.normal(0.0, eps, (DIMM, DIMM))
  flv, fle = orth_norm_vectors(flv, eps)
  all_fle = np.zeros((STEPS, DIMM))
  for i in range(STEPS, 0, -1):
    true[:] = all_true[i-1,:]
    m = finite_time_tangent_using_nonlinear(true, DT, 1)
    flv = np.dot(m.T, flv)
    if (i % orth_int == 0):
      flv, fle = orth_norm_vectors(flv, eps)
      all_fle[i-1,:] = fle[:] / DT
    all_flv[i-1,:,:] = flv[:,:]

  # calculate CLVs i-1 -> i
  all_clv = np.empty((STEPS, DIMM, DIMM))
  for i in range(0, STEPS):
    for k in range(0, DIMM):
      all_clv[i,:,k] = vector_common(all_blv[i,:,:k+1], all_flv[i,:,k:], k, eps)
    # directional continuity
    if (i >= 1):
      m = finite_time_tangent_using_nonlinear(all_true[i-1,:], DT, 1)
      for k in range(0, DIMM):
        clv_approx = np.dot(m, all_clv[i-1,:,k,np.newaxis]).flatten()
        if (np.dot(clv_approx, all_clv[i,:,k]) < 0):
          all_clv[i,:,k] *= -1

  all_true.tofile("data/true.bin")
  all_blv.tofile("data/blv.bin")
  all_ble.tofile("data/ble.bin")
  all_flv.tofile("data/flv.bin")
  all_fle.tofile("data/fle.bin")
  all_clv.tofile("data/clv.bin")

  f = open("data/lyapunov.txt", "w")
  f.write("backward LEs:\n")
  f.write(str_vector(np.mean(all_ble[STEPS//2:,:], axis=0)) + "\n")
  f.write("forward LEs:\n")
  f.write(str_vector(np.mean(all_fle[STEPS//2:,:], axis=0)) + "\n")
  f.write("CLV RMS (column: LVs, row: model grid):\n")
  for i in range (DIMM):
    f.write(str_vector(np.mean(all_clv[STEPS//2:,i,:]**2 / eps**2, axis=0)) + "\n")
  f.close()
  os.system("cat data/lyapunov.txt")
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
  ### All array-like objects in this method are np.ndarray

  # prepare containers
  r = getr()
  h = geth(exp["diag"])
  fcst = np.empty((exp["nmem"], DIMM))
  all_ba = np.empty((STEPS, DIMM, DIMM))
  all_ba[:,:,:] = np.nan
  all_bf = np.empty((STEPS, DIMM, DIMM))
  all_bf[:,:,:] = np.nan
  obs_used = np.empty((STEPS, DIMO))
  obs_used[:,:] = np.nan

  # forecast-analysis cycle
  for i in range(STEP_FREE, STEPS):

    for m in range(0, exp["nmem"]):
      if (exp["couple"] == "strong" or exp["couple"] == "weak"):
        fcst[m,:] = timestep(all_fcst[i-1,m,:], DT)
      elif (exp["couple"] == "none"):
        fcst[m,0:6] = timestep(all_fcst[i-1,m,0:6], DT, 0, 6)
        fcst[m,6:9] = timestep(all_fcst[i-1,m,6:9], DT, 6, 9)

    if (i % exp["aint"] == 0):
      obs_used[i,:] = all_obs[i,:]
      fcst_pre = all_fcst[i-exp["aint"],:,:]

      if (exp["couple"] == "strong"):
        fcst[:,:], all_bf[i,:,:], all_ba[i,:,:] = \
          analyze_one_window(fcst, fcst_pre, all_obs[i,:], h, r, exp)
      elif (exp["couple"] == "weak" or exp["couple"] == "none"):
        dim_atm = 6
        # atmospheric assimilation
        fcst[:, :dim_atm], \
          all_bf[i, :dim_atm, :dim_atm], \
          all_ba[i, :dim_atm, :dim_atm]  \
          = analyze_one_window(fcst[:, :dim_atm], fcst_pre[:, :dim_atm], \
            all_obs[i, :dim_atm], h[:dim_atm, :dim_atm], r[:dim_atm, :dim_atm], exp, 0, 6)
        # oceanic assimilation
        fcst[:, dim_atm:], \
          all_bf[i, dim_atm:, dim_atm:], \
          all_ba[i, dim_atm:, dim_atm:]  \
          = analyze_one_window(fcst[:, dim_atm:], fcst_pre[:, dim_atm:], \
            all_obs[i, dim_atm:], h[dim_atm:, dim_atm:], r[dim_atm:, dim_atm:], exp, 6, 9)

    all_fcst[i,:,:] = fcst[:,:]

  # save to files
  obs_used.tofile("data/%s_obs.bin" % exp["name"])
  all_fcst.tofile("data/%s_cycle.bin" % exp["name"])
  all_bf.tofile("data/%s_covr_back.bin" % exp["name"])
  all_ba.tofile("data/%s_covr_anl.bin" % exp["name"])
  return all_fcst

def analyze_one_window(fcst, fcst_pre, obs, h, r, exp, i_s=0, i_e=DIMM):
  ### here, (dimc = i_e - i_s) unless strongly coupled
  # fcst     <- np.array[nmem, dimc]
  # fcst_pre <- np.array[nmem, dimc]
  # obs      <- np.array[DIMO]
  # h        <- np.array[DIMO, dimc]
  # r        <- np.array[DIMO, DIMO]
  # exp      <- hash
  # i_s      <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e      <- int
  # return1  -> np.array[nmem, dimc]
  # return2  -> np.array[dimc, dimc]
  # return3  -> np.array[dimc, dimc]

  anl = np.empty((exp["nmem"], i_e-i_s))
  bf = np.empty((i_e-i_s, i_e-i_s))
  bf[:,:] = np.nan
  ba = np.empty((i_e-i_s, i_e-i_s))
  ba[:,:] = np.nan

  yo = np.dot(h[:,:], obs[:, np.newaxis])

  if (exp["method"] == "etkf"):
    anl[:,:], bf[:,:], ba[:,:] = \
        etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], exp["inf"], exp["nmem"])
  elif (exp["method"] == "3dvar"):
    anl[0,:] = tdvar(fcst[0,:].T, h[:,:], r[:,:], yo[:,:], i_s, i_e)
  elif (exp["method"] == "4dvar"):
    anl[0,:] = fdvar(fcst_pre[0,:], h[:,:], r[:,:], yo[:,:], exp["aint"], i_s, i_e)

  return anl[:,:], bf[:,:], ba[:,:]

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

def str_vector(arr):
  n = len(arr)
  st = ""
  for i in range(n):
    st += "%11g, " % arr[i]
  return st[:-2]

main()

