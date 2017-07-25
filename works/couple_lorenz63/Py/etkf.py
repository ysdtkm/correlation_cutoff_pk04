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
  if r_local in ["covariance-mean", "correlation-mean", "covariance-rms", \
                 "correlation-rms", "random", "bhhtri-mean", "bhhtri-rms"]:
    if num_yes == None:
      num_yes = 37
    weight_table = choose_weight_order(r_local, num_yes)
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

def choose_weight_order(r_local, odr_max=37):
  # order_table obtained from d265ebb, unit_test.py (diagonal elements prioritized)
  if r_local == "correlation-mean":
    order_table = np.array([
    [ 0,  9, 49, 57, 39, 41, 71, 73, 51],
    [10,  1, 53, 59, 37, 61, 75, 65, 43],
    [50, 54,  2, 45, 55, 67, 79, 69, 77],
    [58, 60, 46,  3, 11, 25, 33, 23, 29],
    [40, 38, 56, 12,  4, 47, 21, 17, 35],
    [42, 62, 68, 26, 48,  5, 31, 27, 19],
    [72, 76, 80, 34, 22, 32,  6, 13, 15],
    [74, 66, 70, 24, 18, 28, 14,  7, 63],
    [52, 44, 78, 30, 36, 20, 16, 64,  8]],
    dtype=np.int32)
  elif r_local == "correlation-rms":
    order_table = np.array([
    [ 0, 11, 17, 65, 47, 67, 63, 59, 53],
    [12,  1, 25, 57, 45, 61, 51, 55, 49],
    [18, 26,  2, 77, 69, 79, 75, 73, 71],
    [66, 58, 78,  3, 13, 21, 43, 31, 33],
    [48, 46, 70, 14,  4, 41, 39, 37, 35],
    [68, 62, 80, 22, 42,  5, 29, 27, 19],
    [64, 52, 76, 44, 40, 30,  6, 15,  9],
    [60, 56, 74, 32, 38, 28, 16,  7, 23],
    [54, 50, 72, 34, 36, 20, 10, 24,  8]],
    dtype=np.int32)
  elif r_local == "covariance-mean":
    order_table = np.array([
    [ 9,  6, 35, 75, 49, 69, 59, 43, 55],
    [ 7,  2, 28, 73, 47, 65, 57, 45, 51],
    [36, 29,  3, 63, 61, 71, 77, 79, 53],
    [76, 74, 64, 34, 32, 39, 41, 30, 22],
    [50, 48, 62, 33, 21, 67, 37, 15, 19],
    [70, 66, 72, 40, 68, 14, 26, 10, 12],
    [60, 58, 78, 42, 38, 27,  8,  4, 24],
    [44, 46, 80, 31, 16, 11,  5,  0, 17],
    [56, 52, 54, 23, 20, 13, 25, 18,  1]],
    dtype=np.int32)
  elif r_local == "covariance-rms":
    order_table = np.array([
    [21, 10, 14, 79, 75, 69, 71, 53, 55],
    [11,  4, 12, 73, 63, 57, 61, 47, 51],
    [15, 13,  7, 77, 67, 65, 59, 43, 49],
    [80, 74, 78, 42, 40, 38, 45, 29, 27],
    [76, 64, 68, 41, 33, 34, 36, 22, 24],
    [70, 58, 66, 39, 35, 26, 31, 18, 16],
    [72, 62, 60, 46, 37, 32, 20,  5,  8],
    [54, 48, 44, 30, 23, 19,  6,  1,  2],
    [56, 52, 50, 28, 25, 17,  9,  3,  0]],
    dtype=np.int32)
  elif r_local == "bhhtri-mean":
    order_table = np.array([
    [ 4,  2, 21, 67, 43, 61, 77, 71, 75],
    [ 3,  0, 15, 65, 37, 57, 76, 72, 73],
    [22, 16,  1, 55, 53, 63, 78, 80, 74],
    [68, 66, 56, 20, 18, 26, 69, 47, 40],
    [44, 38, 54, 19, 10, 59, 49, 32, 39],
    [62, 58, 64, 27, 60,  7, 46, 30, 31],
    [52, 51, 70, 28, 23, 14, 29, 24, 41],
    [33, 34, 79, 17,  8,  5, 25, 12, 35],
    [50, 45, 48, 11,  9,  6, 42, 36, 13]],
    dtype=np.int32)
  elif r_local == "bhhtri-rms":
    order_table = np.array([
    [10,  2,  6, 67, 61, 56, 80, 76, 77],
    [ 3,  0,  4, 59, 48, 41, 79, 73, 75],
    [ 7,  5,  1, 64, 52, 50, 78, 71, 74],
    [68, 60, 65, 29, 27, 25, 72, 66, 63],
    [62, 49, 53, 28, 19, 20, 70, 54, 55],
    [57, 42, 51, 26, 21, 13, 69, 46, 43],
    [58, 45, 44, 31, 24, 18, 47, 33, 35],
    [39, 32, 30, 15, 11,  9, 34, 17, 22],
    [40, 38, 37, 14, 12,  8, 36, 23, 16]],
    dtype=np.int32)
  elif r_local == "random":
    order_table = np.array([
    [68, 54, 40, 11, 48, 79, 66, 34, 73],
    [53, 43, 14, 42, 26, 13,  1, 35, 64],
    [52, 21, 28, 27, 23, 44, 62, 16, 70],
    [74,  5,  0, 10,  2,  7, 60, 65,  6],
    [19, 37, 49, 69, 30, 45,  4, 63, 78],
    [20, 18, 51, 38, 32, 72, 33,  9, 59],
    [22, 55, 41, 39, 29, 61, 57,  8, 58],
    [67, 75, 77, 76, 71, 47, 36, 50, 56],
    [12, 46,  3, 31, 80, 15, 24, 25, 17]],
    dtype=np.int32)
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

choose_weight_order("covariance-rms")
