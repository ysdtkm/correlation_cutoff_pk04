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
  np.random.seed((10**9+7)*1)
  nature = exec_nature()

  np.random.seed((10**9+7)*3)
  obs = exec_obs(nature)

  for settings in EXPLIST:
    np.random.seed((10**9+7)*4)
    free = exec_free_run(settings)
    anl  = exec_assim_cycle(settings, free, obs)
    exec_deterministic_fcst(settings, anl)

def exec_nature():
  # return   -> np.array[STEPS, DIMM]

  all_true = np.empty((STEPS, DIMM))
  true = np.random.randn(DIMM) * FERR_INI

  # forward integration i-1 -> i
  for i in range(0, STEPS):
    true[:] = timestep(true[:], DT)
    all_true[i,:] = true[:]
  all_true.tofile("data/true.bin")

  np.random.seed((10**9+7)*2)
  if Calc_lv:
    all_blv, all_ble = calc_blv(all_true)
    all_flv, all_fle = calc_flv(all_true)
    all_clv          = calc_clv(all_true, all_blv, all_flv)
    calc_fsv(all_true)
    calc_isv(all_true)
    write_lyapunov_exponents(all_ble, all_fle, all_clv)

  return all_true

def exec_obs(nature):
  # nature   <- np.array[STEPS, DIMM]
  # return   -> np.array[STEPS, DIMO]

  all_obs = np.empty((STEPS, DIMO))
  for i in range(0, STEPS):
    if DIMM == 9:
      all_obs[i,:6] = nature[i,:6] + np.random.randn(6) * OERR_A
      all_obs[i,6:] = nature[i,6:] + np.random.randn(3) * OERR_O
    else:
      all_obs[i,:] = nature[i,:] + np.random.randn(DIMO) * OERR_A

  all_obs.tofile("data/obs.bin")
  return all_obs

def exec_free_run(settings):
  # settings <- hash
  # return   -> np.array[STEPS, nmem, DIMM]

  free_run = np.empty((STEPS, settings["nmem"], DIMM))
  for m in range(0, settings["nmem"]):
    free_run[0,m,:] = np.random.randn(DIMM) * FERR_INI
    for i in range(1, STEP_FREE):
      free_run[i,m,:] = timestep(free_run[i-1,m,:], DT)
  return free_run

def exec_assim_cycle(settings, all_fcst, all_obs):
  # settings <- hash
  # all_fcst <- np.array[STEPS, nmem, DIMM]
  # all_obs  <- np.array[STEPS, DIMO]
  # return   -> np.array[STEPS, nmem, DIMM]

  # prepare containers
  r = getr()
  h = geth()
  fcst = np.empty((settings["nmem"], DIMM))
  all_ba = np.empty((STEPS, DIMM, DIMM))
  all_ba[:,:,:] = np.nan
  all_bf = np.empty((STEPS, DIMM, DIMM))
  all_bf[:,:,:] = np.nan
  obs_used = np.empty((STEPS, DIMO))
  obs_used[:,:] = np.nan

  all_inflation = np.empty((STEPS, 3))
  all_inflation[:,:] = np.nan
  if settings["method"] == "etkf" and (settings["rho"] == "adaptive" or settings["rho"] == "adaptive_each"):
    obj_adaptive = init_etkf_adaptive_inflation()
  else:
    obj_adaptive = None

  # forecast-analysis cycle
  try:
    for i in range(STEP_FREE, STEPS):
      if settings["couple"] == "none" and settings["bc"] == "climatology":
        persis_bc = None
      elif settings["couple"] == "none" and settings["bc"] == "independent":
        persis_bc = all_obs[i-1,:].copy()
      else: # persistence BC
        persis_bc = np.mean(all_fcst[i-1,:,:], axis=0)

      for m in range(0, settings["nmem"]):
        if (settings["couple"] == "strong" or settings["couple"] == "weak"):
          fcst[m,:] = timestep(all_fcst[i-1,m,:], DT)
        elif (settings["couple"] == "none"):
          fcst[m,0:6] = timestep(all_fcst[i-1,m,0:6], DT, 0, 6, persis_bc)
          fcst[m,6:9] = timestep(all_fcst[i-1,m,6:9], DT, 6, 9, persis_bc)

      if (i % AINT == 0):
        obs_used[i,:] = all_obs[i,:]
        fcst_pre = all_fcst[i-AINT,:,:].copy()

        if (settings["couple"] == "strong"):
          fcst[:,:], all_bf[i,:,:], all_ba[i,:,:], obj_adaptive = \
            analyze_one_window(fcst, fcst_pre, all_obs[i,:], h, r, settings, obj_adaptive)
        elif (settings["couple"] == "weak" or settings["couple"] == "none"):
          dim_atm = 6
          # atmospheric assimilation
          fcst[:, :dim_atm], all_bf[i, :dim_atm, :dim_atm], all_ba[i, :dim_atm, :dim_atm], obj_adaptive \
              = analyze_one_window(fcst[:, :dim_atm], fcst_pre[:, :dim_atm],
                                   all_obs[i, :dim_atm], h[:dim_atm, :dim_atm],
                                   r[:dim_atm, :dim_atm], settings, obj_adaptive, 0, 6, persis_bc)
          # oceanic assimilation
          fcst[:, dim_atm:], all_bf[i, dim_atm:, dim_atm:], all_ba[i, dim_atm:, dim_atm:], obj_adaptive  \
              = analyze_one_window(fcst[:, dim_atm:], fcst_pre[:, dim_atm:],
                                   all_obs[i, dim_atm:], h[dim_atm:, dim_atm:],
                                   r[dim_atm:, dim_atm:], settings, obj_adaptive, 6, 9, persis_bc)

      all_fcst[i,:,:] = fcst[:,:]
      if settings["method"] == "etkf" and (settings["rho"] == "adaptive" or settings["rho"] == "adaptive_each"):
        all_inflation[i,:] = obj_adaptive[0,:]
  except (ValueError, RuntimeWarning) as e:
    import traceback
    print("Analysis cycle diverged: %s" % e)
    print("Settings: ", settings)
    traceback.print_exc()
    sys.exit(3) # ttk

  # save to files
  obs_used.tofile("data/%s_obs.bin" % settings["name"])
  all_fcst.tofile("data/%s_cycle.bin" % settings["name"])
  all_bf.tofile("data/%s_covr_back.bin" % settings["name"])
  all_ba.tofile("data/%s_covr_anl.bin" % settings["name"])
  if settings["method"] == "etkf" and (settings["rho"] == "adaptive" or settings["rho"] == "adaptive_each"):
    all_inflation.tofile("data/%s_inflation.bin" % settings["name"])

  return all_fcst

def analyze_one_window(fcst, fcst_pre, obs, h, r, settings, obj_adaptive, i_s=0, i_e=DIMM, bc=None):
  # fcst     <- np.array[nmem, dimc]
  # fcst_pre <- np.array[nmem, dimc]
  # obs      <- np.array[DIMO]
  # h        <- np.array[DIMO, dimc]
  # r        <- np.array[DIMO, DIMO]
  # settings <- hash
  # obj_adaptive
  #          <- object created by etkf/init_etkf_adaptive_inflation(), or None
  # i_s      <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e      <- int
  # bc       <- np.array[DIMM]
  # return1  -> np.array[nmem, dimc]
  # return2  -> np.array[dimc, dimc]
  # return3  -> np.array[dimc, dimc]

  anl = np.empty((settings["nmem"], i_e-i_s))
  bf = np.empty((i_e-i_s, i_e-i_s))
  ba = np.empty((i_e-i_s, i_e-i_s))
  bf[:,:] = np.nan
  ba[:,:] = np.nan

  yo = np.dot(h[:,:], obs[:, np.newaxis])

  if (settings["method"] == "etkf"):
    anl[:,:], bf[:,:], ba[:,:], obj_adaptive = \
        etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], settings["rho"],
             settings["nmem"], obj_adaptive, True, settings["r_local"], settings.get("num_yes"))
  elif (settings["method"] == "3dvar"):
    anl[0,:] = tdvar(fcst[0,:].T, h[:,:], r[:,:], yo[:,:], i_s, i_e, settings["amp_b"])
  elif (settings["method"] == "4dvar"):
    anl[0,:] = fdvar(fcst_pre[0,:], h[:,:], r[:,:], yo[:,:], AINT, i_s, i_e, settings["amp_b"], bc)
  else:
    raise Exception("analysis method mis-specified: %s" % settings["method"])

  return anl[:,:], bf[:,:], ba[:,:], obj_adaptive

def exec_deterministic_fcst(settings, anl):
  # settings <- hash
  # anl      <- np.array[STEPS, nmem, DIMM]
  # return   -> np.array[STEPS, FCST_LT, DIMM]

  if FCST_LT == 0:
    return 0

  fcst_all = np.empty((STEPS, FCST_LT, DIMM))
  for i in range(STEP_FREE, STEPS):
    if (i % AINT == 0):
      fcst_all[i,0,:] = np.mean(anl[i,:,:], axis=0)
      for lt in range(1, FCST_LT):
        # todo: uncoupled forecast
        fcst_all[i,lt,:] = timestep(fcst_all[i-1,lt,:], DT)
  fcst_all.tofile("data/%s_fcst.bin" % settings["name"])
  return 0

if __name__ == "__main__":
  main()

