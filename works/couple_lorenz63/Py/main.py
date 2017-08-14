#!/usr/bin/env python

import os
import numpy as np
from const import *
import model, etkf, tdvar, fdvar, vectors

def main():
  np.random.seed((10**9+7)*1)
  nature = exec_nature()

  np.random.seed((10**9+7)*3)
  obs = exec_obs(nature)

  for settings in EXPLIST:
    print("Analysis cycle: %s" % settings["name"])
    np.random.seed((10**9+7)*4)
    free = exec_free_run(settings)
    anl  = exec_assim_cycle(settings, free, obs)
    exec_deterministic_fcst(settings, anl)

def exec_nature():
  # return   -> np.array[STEPS, N_MODEL]

  all_true = np.empty((STEPS, N_MODEL))
  true = np.random.randn(N_MODEL) * FERR_INI

  # forward integration i-1 -> i
  for i in range(0, STEPS):
    true[:] = model.timestep(true[:], DT)
    all_true[i,:] = true[:]
  all_true.tofile("data/true.bin")

  np.random.seed((10**9+7)*2)
  if Calc_lv:
    all_blv, all_ble = vectors.calc_blv(all_true)
    all_flv, all_fle = vectors.calc_flv(all_true)
    all_clv          = vectors.calc_clv(all_true, all_blv, all_flv)
    vectors.calc_fsv(all_true)
    vectors.calc_isv(all_true)
    vectors.write_lyapunov_exponents(all_ble, all_fle, all_clv)

  return all_true

def exec_obs(nature):
  ### note: currently this method cannot handle non-diagonal element of R
  # nature   <- np.array[STEPS, N_MODEL]
  # return   -> np.array[STEPS, P_OBS]

  all_obs = np.empty((STEPS, P_OBS))
  h = geth()
  r = getr()
  for i in range(0, STEPS):
    all_obs[i,:] = h.dot(nature[i,:]) + np.random.randn(P_OBS) * r.diagonal()**0.5

  all_obs.tofile("data/obs.bin")
  return all_obs

def exec_free_run(settings):
  # settings <- hash
  # return   -> np.array[STEPS, nmem, N_MODEL]

  free_run = np.empty((STEPS, settings["nmem"], N_MODEL))
  for m in range(0, settings["nmem"]):
    free_run[0,m,:] = np.random.randn(N_MODEL) * FERR_INI
    for i in range(1, STEP_FREE):
      free_run[i,m,:] = model.timestep(free_run[i-1,m,:], DT)
  return free_run

def exec_assim_cycle(settings, all_fcst, all_obs):
  # settings <- hash
  # all_fcst <- np.array[STEPS, nmem, N_MODEL]
  # all_obs  <- np.array[STEPS, P_OBS]
  # return   -> np.array[STEPS, nmem, N_MODEL]

  n_atm = 6
  p_atm = 6 # ttk, It should be specified in const.py

  # prepare containers
  r = getr()
  h = geth()
  fcst = np.empty((settings["nmem"], N_MODEL))
  all_ba = np.empty((STEPS, N_MODEL, N_MODEL))
  all_ba[:,:,:] = np.nan
  all_bf = np.empty((STEPS, N_MODEL, N_MODEL))
  all_bf[:,:,:] = np.nan
  obs_used = np.empty((STEPS, P_OBS))
  obs_used[:,:] = np.nan

  all_inflation = np.empty((STEPS, 3))
  all_inflation[:,:] = np.nan
  if settings["method"] == "etkf" and (settings["rho"] == "adaptive" or settings["rho"] == "adaptive_each"):
    obj_adaptive = etkf.init_etkf_adaptive_inflation()
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
          fcst[m,:] = model.timestep(all_fcst[i-1,m,:], DT)
        elif (settings["couple"] == "none"):
          fcst[m, :n_atm] = model.timestep(all_fcst[i-1, m, :n_atm], DT, 0, n_atm, persis_bc)
          fcst[m, n_atm:] = model.timestep(all_fcst[i-1, m, n_atm:], DT, n_atm, N_MODEL, persis_bc)

      if (i % AINT == 0):
        obs_used[i,:] = all_obs[i,:]
        fcst_pre = all_fcst[i-AINT,:,:].copy()

        if (settings["couple"] == "strong"):
          fcst[:,:], all_bf[i,:,:], all_ba[i,:,:], obj_adaptive = \
            analyze_one_window(fcst, fcst_pre, all_obs[i,:], h, r, settings, obj_adaptive)
        elif (settings["couple"] == "weak" or settings["couple"] == "none"):
          # atmospheric assimilation
          fcst[:, :n_atm], all_bf[i, :n_atm, :n_atm], all_ba[i, :n_atm, :n_atm], obj_adaptive \
              = analyze_one_window(fcst[:, :n_atm], fcst_pre[:, :n_atm],
                                   all_obs[i, :p_atm], h[:p_atm, :n_atm],
                                   r[:p_atm, :p_atm], settings, obj_adaptive, 0, n_atm, persis_bc)
          # oceanic assimilation
          fcst[:, n_atm:], all_bf[i, n_atm:, n_atm:], all_ba[i, n_atm:, n_atm:], obj_adaptive  \
              = analyze_one_window(fcst[:, n_atm:], fcst_pre[:, n_atm:],
                                   all_obs[i, p_atm:], h[p_atm:, n_atm:],
                                   r[p_atm:, p_atm:], settings, obj_adaptive, n_atm, N_MODEL, persis_bc)

      all_fcst[i,:,:] = fcst[:,:]
      if settings["method"] == "etkf" and (settings["rho"] == "adaptive" or settings["rho"] == "adaptive_each"):
        all_inflation[i,:] = obj_adaptive[0,:]
  except (np.linalg.LinAlgError, ValueError) as e:
    import traceback
    print("")
    print("ANALYSIS CYCLE DIVERGED: %s" % e)
    print("Settings: ", settings)
    print("This experiment is terminated (see error traceback below). Continue on next experiments.")
    print("")
    traceback.print_exc()
    print("")

  # save to files
  obs_used.tofile("data/%s_obs.bin" % settings["name"])
  all_fcst.tofile("data/%s_cycle.bin" % settings["name"])
  all_bf.tofile("data/%s_covr_back.bin" % settings["name"])
  all_ba.tofile("data/%s_covr_anl.bin" % settings["name"])
  if settings["method"] == "etkf" and (settings["rho"] == "adaptive" or settings["rho"] == "adaptive_each"):
    all_inflation.tofile("data/%s_inflation.bin" % settings["name"])

  return all_fcst

def analyze_one_window(fcst, fcst_pre, obs, h, r, settings, obj_adaptive, i_s=0, i_e=N_MODEL, bc=None):
  # fcst     <- np.array[nmem, dimc]
  # fcst_pre <- np.array[nmem, dimc]
  # obs      <- np.array[pc_obs]
  # h        <- np.array[pc_obs, dimc]
  # r        <- np.array[pc_obs, pc_obs]
  # settings <- hash
  # obj_adaptive
  #          <- object created by etkf/init_etkf_adaptive_inflation(), or None
  # i_s      <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e      <- int
  # bc       <- np.array[N_MODEL]
  # return1  -> np.array[nmem, dimc]
  # return2  -> np.array[dimc, dimc]
  # return3  -> np.array[dimc, dimc]

  anl = np.empty((settings["nmem"], i_e-i_s))
  bf = np.empty((i_e-i_s, i_e-i_s))
  ba = np.empty((i_e-i_s, i_e-i_s))
  bf[:,:] = np.nan
  ba[:,:] = np.nan

  yo = obs[:, np.newaxis]

  if (settings["method"] == "etkf"):
    anl[:,:], bf[:,:], ba[:,:], obj_adaptive = \
        etkf.etkf(fcst[:,:], h[:,:], r[:,:], yo[:,:], settings["rho"],
             settings["nmem"], obj_adaptive, True, settings["r_local"], settings.get("num_yes"))
  elif (settings["method"] == "3dvar"):
    anl[0,:] = tdvar.tdvar(fcst[0,:].T, h[:,:], r[:,:], yo[:,:], i_s, i_e, settings["amp_b"])
  elif (settings["method"] == "4dvar"):
    anl[0,:] = fdvar.fdvar(fcst_pre[0,:], h[:,:], r[:,:], yo[:,:], AINT, i_s, i_e, settings["amp_b"], bc)
  else:
    raise Exception("analysis method mis-specified: %s" % settings["method"])

  return anl[:,:], bf[:,:], ba[:,:], obj_adaptive

def exec_deterministic_fcst(settings, anl):
  # settings <- hash
  # anl      <- np.array[STEPS, nmem, N_MODEL]
  # return   -> np.array[STEPS, FCST_LT, N_MODEL]

  if FCST_LT == 0:
    return 0

  fcst_all = np.empty((STEPS, FCST_LT, N_MODEL))
  for i in range(STEP_FREE, STEPS):
    if (i % AINT == 0):
      fcst_all[i,0,:] = np.mean(anl[i,:,:], axis=0)
      for lt in range(1, FCST_LT):
        # todo: uncoupled forecast
        fcst_all[i,lt,:] = model.timestep(fcst_all[i-1,lt,:], DT)
  fcst_all.tofile("data/%s_fcst.bin" % settings["name"])
  return 0

if __name__ == "__main__":
  main()

