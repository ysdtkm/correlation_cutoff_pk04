#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import *
import stats_const

def etkf(fcst, h_nda, r_nda, yo_nda, rho_in, nmem, obj_adaptive, localization=False, r_local="", num_yes=None):
  # fcst   <- np.array[nmem, dimc]
  # h_nda  <- np.array[DIMO, dimc]
  # r_nda  <- np.array[DIMO, DIMO]
  # yo_nda <- np.array[DIMO, 1]
  # rho    <- float
  # nmem   <- int
  # obj_adaptive
  #        <- object created by etkf/init_etkf_adaptive_inflation(), or None
  # r_local
  #        <- (string): localization pattern of R
  # return -> np.array[dimc, nmem], np.array[dimc, dimc], np.array[dimc, dimc]

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  dimc = fcst.shape[1]

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
    xai = np.matrix(np.zeros((dimc, nmem)))

    if rho_in == "adaptive":
      delta_this = obtain_delta_this_step(yo, yb, ybpt, r, nmem, True)
      obj_adaptive = update_adaptive_inflation(obj_adaptive, delta_this)
    elif rho_in == "adaptive_each":
      delta_this = obtain_delta_this_step(yo, yb, ybpt, r, nmem, False)
      obj_adaptive = update_adaptive_inflation(obj_adaptive, delta_this)

    for j in range(dimc):
      # step 3
      localization_weight = obtain_localization_weight(dimc, j, r_local, num_yes)
      yol = yo[:,:].copy()
      ybl = yb[:,:].copy()
      ybptl = ybpt[:,:].copy()
      xfl = xf[j,:].copy()
      xfptl = xfpt[j,:].copy()
      rl = r[:,:].copy()

      if rho_in == "adaptive" or rho_in == "adaptive_each":
        component = [0,0,0,1,1,1,2,2,2]
        rho = obj_adaptive[0,component[j]]
      else:
        rho = rho_in

      # step 4-9
      cl = ybptl.T * np.asmatrix(rl.I.A * localization_weight.A)
      pal = (((nmem-1.0)/rho) * I_mm + cl * ybptl).I
      waptl = np.matrix(np.real(sqrtm((nmem-1.0) * pal)))
      wal = pal * cl * (yol - ybl)
      xail = xfl * I_1m + xfptl * (wal * I_1m + waptl)
      xai[j,:] = xail[:,:]

    xapt = xai - np.mean(xai[:,:], axis=1) * I_1m
    return np.real(xai.T.A), (xfpt * xfpt.T).A, (xapt * xapt.T).A, obj_adaptive

  else:

    if rho_in == "adaptive" or rho_in == "adaptive_each":
      raise Exception("non-localized ETKF cannot handle adaptive inflation")
    else:
      rho = rho_in

    pa   = (((nmem-1.0)/rho) * I_mm + ybpt.T * r.I * ybpt).I
    wam  = np.matrix(sqrtm((nmem-1.0) * pa))
    wa   = pa * ybpt.T * r.I * (yo - yb)
    xapt = (xfm - xf * I_1m) * wam
    xa   = xf + xfm * wa
    xam  = xapt + xa * I_1m
    return np.real(xam.T.A), (xfpt * xfpt.T).A, (xapt * xapt.T).A, obj_adaptive

def obtain_localization_weight(dimc, j, r_local, num_yes):
  # dimc   <- int : cimension of analyzed component
  # j      <- int : index of analyzed grid
  # r_local (string): localization pattern of R
  # return -> np.matrix : R-inverse localizaiton weight matrix

  localization_weight = np.ones((dimc, dimc))

  if DIMM != 9:
    return localization_weight

  if dimc == DIMM: # strongly coupled
    weight_table = get_weight_table(r_local, num_yes)
    for iy in range(dimc):
      localization_weight[iy, :] *= weight_table[iy, j]
      localization_weight[:, iy] *= weight_table[iy, j]

  elif dimc == 3:
    pass

  elif dimc == 6:
    pass

  return np.asmatrix(localization_weight)

def get_weight_table(r_local, num_yes):
  # return weight_table[iy, ix] : weight of iy-th obs for ix-th grid

  if r_local in ["covariance-mean", "correlation-mean", "covariance-rms",
                 "correlation-rms", "random", "BHHtRi-mean", "BHHtRi-rms",
                 "covariance-clim", "correlation-clim"]:
    if num_yes == None:
      num_yes = 37
    weight_table = weight_based_on_stats(r_local, num_yes)
  elif r_local == "dynamical": # a38p35
    weight_table = np.array([
      [1,1,1,  1,0,0,  0,0,0], [1,1,1,  0,1,0,  0,0,0], [1,1,1,  0,0,0,  0,0,0],
      [1,0,0,  1,1,1,  1,0,0], [0,1,0,  1,1,1,  0,1,0], [0,0,0,  1,1,1,  0,0,1],
      [0,0,0,  1,0,0,  1,1,1], [0,0,0,  0,1,0,  1,1,1], [0,0,0,  0,0,1,  1,1,1]], dtype=np.float64)
  else:
    weight_table_small = {
      "individual":        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
      "atmos_coupling":    np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
      "enso_coupling":     np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
      "atmos_sees_ocean":  np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
      "trop_sees_ocean":   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
      "ocean_sees_atmos":  np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
      "ocean_sees_trop":   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
      "full":              np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      "adjacent":          np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
    }
    weight_table = np.ones((DIMM, DIMM))
    for iyc in range(3):
      for ixc in range(3):
        weight_table[iyc*3:iyc*3+3, ixc*3:ixc*3+3] = \
          weight_table_small[r_local][iyc, ixc]
  return weight_table

def weight_based_on_stats(r_local, odr_max=37):
  if DIMM != 9:
    raise Exception("stats_weight_order() is only for 9-variable PK04 model")

  order_table = stats_const.stats_order(r_local)
  return np.float64(order_table < odr_max)

def init_etkf_adaptive_inflation():
  # return : np.array([[delta_extra, delta_trop, delta_ocean],
  #                    [var_extra, var_trop, var_ocean]])

  if DIMM != 9:
    raise Exception("adaptive inflation is only for 9-variable coupled model")

  obj_adaptive = np.array([[1.05, 1.05, 1.05],
                           [1.0,  1.0,  1.0]])
  return obj_adaptive

def update_adaptive_inflation(obj_adaptive, delta_this_step):
  # obj_adaptive (in, out):
  #   np.array([[delta_extra, delta_trop, delta_ocean],
  #             [var_extra, var_trop, var_ocean]])
  # delta_this_step:
  #   np.array([delta_extra, delta_trop, delta_ocean])

  vo = 1.0
  kappa = 1.01

  # limit delta_this_step
  delta_max = np.array([1.2, 1.2, 1.2])
  delta_min = np.array([0.9, 0.9, 0.9])
  delta_this_step = np.max(np.row_stack((delta_min, delta_this_step)), axis=0)
  delta_this_step = np.min(np.row_stack((delta_max, delta_this_step)), axis=0)

  vf = obj_adaptive[1,:].copy() * kappa
  delta_new = (obj_adaptive[0,:] * vo + delta_this_step[:] * vf[:]) / (vo + vf[:])
  va = (vf[:] * vo) / (vf[:] + vo)

  obj_adaptive[0,:] = delta_new[:]
  obj_adaptive[1,:] = va[:]

  return obj_adaptive

def obtain_delta_this_step(yo, yb, ybpt, r, nmem, common):

  if DIMM != 9 or yo.shape[0] != 9:
    raise Exception("this method is only used when DIMM == 9")

  delta = np.empty(3)

  if common:
    dob = yo - yb
    delta[:] = (dob.T.dot(dob) - np.trace(r)) / np.trace(ybpt.dot(ybpt.T) / (nmem-1))
  else:
    for i in range(3):
      yol = yo[i*3:i*3+3]
      ybl = yb[i*3:i*3+3]
      ybptl = ybpt[i*3:i*3+3,:]
      rl = r[i*3:i*3+3,i*3:i*3+3]
      dob = yol - ybl
      delta[i] = (dob.T.dot(dob) - np.trace(rl)) / np.trace(ybptl.dot(ybptl.T) / (nmem-1))
  return delta

