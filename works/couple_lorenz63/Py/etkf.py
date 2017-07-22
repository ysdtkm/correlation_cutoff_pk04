#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import *

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

  # if r_local == "dynamical": # a38p35
  #   weight_table = np.array([
  #     [1,1,1,  1,0,0,  0,0,0], [1,1,1,  0,1,0,  0,0,0], [1,1,1,  0,0,0,  0,0,0],
  #     [1,0,0,  1,1,1,  1,0,0], [0,1,0,  1,1,1,  0,1,0], [0,0,0,  1,1,1,  0,0,1],
  #     [0,0,0,  1,0,0,  1,1,1], [0,0,0,  0,1,0,  1,1,1], [0,0,0,  0,0,1,  1,1,1]], dtype=np.float64)
  # elif r_local == "rms_correlation": # commit:9104078
  #   weight_table = np.array([
  #     [1,1,1,  0,0,0,  0,0,0], [1,1,1,  0,0,0,  0,0,0], [1,1,1,  0,0,0,  0,0,0],
  #     [0,0,0,  1,1,1,  0,0,1], [0,0,0,  1,1,0,  1,0,1], [0,0,0,  1,0,1,  1,1,1],
  #     [0,0,0,  0,1,1,  1,1,1], [0,0,0,  0,0,1,  1,1,1], [0,0,0,  1,1,1,  1,1,1]], dtype=np.float64)
  # elif r_local == "rms_covariance":
  #   weight_table = np.array([
  #     [1,1,1,  0,0,0,  0,0,0], [1,1,1,  0,0,0,  0,0,0], [1,1,1,  0,0,0,  0,0,0],
  #     [0,0,0,  1,1,0,  0,1,1], [0,0,0,  1,1,0,  0,1,1], [0,0,0,  0,0,1,  1,1,1],
  #     [0,0,0,  0,0,1,  1,1,1], [0,0,0,  1,1,1,  1,1,1], [0,0,0,  1,1,1,  1,1,1]], dtype=np.float64)
  # elif r_local == "random":
  #   weight_table = np.array([
  #     [1,0,0,  1,0,0,  0,0,0], [0,1,1,  0,1,1,  1,0,0], [0,1,1,  1,1,0,  0,1,0],
  #     [1,0,1,  1,1,1,  0,0,1], [0,1,1,  1,1,0,  1,0,0], [0,1,0,  1,0,1,  0,1,0],
  #     [0,1,0,  0,1,0,  1,1,0], [0,0,1,  0,0,1,  1,1,0], [0,0,0,  1,0,0,  0,0,1]], dtype=np.float64)
  if r_local in ["dynamical", "covariance-mean", "correlation-mean", "covariance-rms", "correlation-rms", "random", "bhhtri"]:
    if num_yes == None:
      num_yes = 37
    weight_table = choose_weight_order(r_local, num_yes)
  else:
    weight_table_small = {
      "individual":        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
      "extra_trop_couple": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
      "trop_ocean_couple": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
      "atmos_sees_ocean":  np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
      "trop_sees_ocean":   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
      "ocean_sees_atmos":  np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
      "ocean_sees_trop":   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
      "full":              np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      "see_adjacent":      np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
    }
    weight_table = np.ones((DIMM, DIMM))
    for iyc in range(3):
      for ixc in range(3):
        weight_table[iyc*3:iyc*3+3, ixc*3:ixc*3+3] = \
          weight_table_small[r_local][iyc, ixc]
  return weight_table

def choose_weight_order(r_local, odr_max=37):
  # order_table obtained from d265ebb, unit_test.py (diagonal elements prioritized)
  if r_local == "correlation-mean":
    order_table = np.array([
    [ 0,  9, 67, 41, 33, 73, 47, 55, 37],
    [ 9,  1, 39, 43, 31, 63, 49, 59, 35],
    [67, 39,  2, 71, 65, 57, 53, 45, 51],
    [41, 43, 71,  3, 11, 79, 27, 23, 25],
    [33, 31, 65, 11,  4, 69, 21, 17, 29],
    [73, 63, 57, 79, 69,  5, 75, 77, 19],
    [47, 49, 53, 27, 21, 75,  6, 13, 15],
    [55, 59, 45, 23, 17, 77, 13,  7, 61],
    [37, 35, 51, 25, 29, 19, 15, 61,  8]], dtype=np.int32)
  elif r_local == "correlation-rms":
    order_table = np.array([
    [ 0, 11, 17, 65, 47, 67, 63, 59, 53],
    [11,  1, 25, 57, 45, 61, 51, 55, 49],
    [17, 25,  2, 77, 69, 79, 75, 73, 71],
    [65, 57, 77,  3, 13, 21, 43, 31, 33],
    [47, 45, 69, 13,  4, 41, 39, 37, 35],
    [67, 61, 79, 21, 41,  5, 29, 27, 19],
    [63, 51, 75, 43, 39, 29,  6, 15,  9],
    [59, 55, 73, 31, 37, 27, 15,  7, 23],
    [53, 49, 71, 33, 35, 19,  9, 23,  8]], dtype=np.int32)
  elif r_local == "covariance-mean":
    order_table = np.array([
    [ 9,  6, 27, 43, 35, 51, 61, 69, 39],
    [ 6,  2, 20, 41, 33, 55, 63, 67, 37],
    [27, 20,  3, 57, 59, 49, 47, 45, 65],
    [43, 41, 57, 26, 24, 71, 31, 22, 18],
    [35, 33, 59, 24, 17, 53, 29, 13, 15],
    [51, 55, 49, 71, 53, 12, 73, 79, 10],
    [61, 63, 47, 31, 29, 73,  8,  4, 75],
    [69, 67, 45, 22, 13, 79,  4,  0, 77],
    [39, 37, 65, 18, 15, 10, 75, 77,  1]], dtype=np.int32)
  elif r_local == "covariance-rms":
    order_table = np.array([
    [21, 10, 14, 79, 75, 69, 71, 53, 55],
    [10,  4, 12, 73, 63, 57, 61, 47, 51],
    [14, 12,  7, 77, 67, 65, 59, 43, 49],
    [79, 73, 77, 42, 40, 38, 45, 29, 27],
    [75, 63, 67, 40, 33, 34, 36, 22, 24],
    [69, 57, 65, 38, 34, 26, 31, 18, 16],
    [71, 61, 59, 45, 36, 31, 20,  5,  8],
    [53, 47, 43, 29, 22, 18,  5,  1,  2],
    [55, 51, 49, 27, 24, 16,  8,  2,  0]], dtype=np.int32)
  elif r_local == "bhhtri": # ttk
    order_table = np.array([
    [ 4,  2, 19, 40, 32, 56, 66, 75, 37],
    [ 2,  0, 13, 38, 28, 60, 67, 74, 34],
    [19, 13,  1, 62, 64, 54, 53, 46, 68],
    [40, 38, 62, 18, 16, 77, 24, 15, 10],
    [32, 28, 64, 16,  9, 58, 21,  7,  8],
    [56, 60, 54, 77, 58,  6, 79, 80,  5],
    [66, 67, 53, 24, 21, 79, 25, 22, 70],
    [75, 74, 46, 15,  7, 80, 22, 11, 72],
    [37, 34, 68, 10,  8,  5, 70, 72, 12]], dtype=np.int32)
  elif r_local == "random":
    order_table = np.array([
    [68, 54, 40, 11, 48, 79, 66, 34, 73],
    [54, 43, 14, 42, 26, 13,  1, 35, 64],
    [40, 14, 28, 27, 23, 44, 62, 16, 70],
    [11, 42, 27, 10,  2,  7, 60, 65,  6],
    [48, 26, 23,  2, 30, 45,  4, 63, 78],
    [79, 13, 44,  7, 45, 72, 33,  9, 59],
    [66,  1, 62, 60,  4, 33, 57,  8, 58],
    [34, 35, 16, 65, 63,  9,  8, 50, 56],
    [73, 64, 70,  6, 78, 59, 58, 56, 17]], dtype=np.int32)
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
  kappa = 1.03 # ttk

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

choose_weight_order("covariance-rms")
