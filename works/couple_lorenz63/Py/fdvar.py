#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin, fmin_bfgs
from const import *
import stats_const, model

def fdvar(fcst_0, h, r, yo, aint, i_s, i_e, amp_b, bc=None):
  # fcst_0 <- np.array[dimc]       : first guess at beginning of window
  # h      <- np.array[pc_obs, dimc] : observation operator
  # r      <- np.array[pc_obs, pc_obs] : observation error covariance
  # yo     <- np.array[pc_obs, 1]    : observation
  # aint   <- int                  : assimilation interval
  # i_s    <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e    <- int
  # amp_b  <- float
  # bc     <- np.array[N_MODEL]       : boundary condition if needed
  # return -> np.array[dimc]       : assimilated field

  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time

  try:
    anl_0 = np.copy(fcst_0)
    anl_0 = fmin_bfgs(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint, i_s, i_e, amp_b, bc), disp=False)
    # anl_0 = fmin_bfgs(fdvar_2j, anl_0, fprime=fdvar_2j_deriv, args=(fcst_0, h, r, yo, aint, i_s, i_e, amp_b, bc))
  except:
    print("Method fmin_bfgs failed to converge. Use fmin for this step instead.")
    anl_0 = np.copy(fcst_0)
    anl_0 = fmin(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint, i_s, i_e, amp_b, bc), disp=False)

  anl_1 = np.copy(anl_0)
  for i in range(0, aint):
    anl_1 = model.timestep(anl_1, DT, i_s, i_e, bc)
  return anl_1.T

def fdvar_analytical(fcst_0_nda, h_nda, r_nda, yo_nda, aint, i_s, i_e, amp_b, bc=None):
  # fcst_0_nda <- np.array[dimc]       : first guess at beginning of window
  # h_nda      <- np.array[pc_obs, dimc] : observation operator
  # r_nda      <- np.array[pc_obs, pc_obs] : observation error covariance
  # yo_nda     <- np.array[pc_obs, 1]    : observation
  # aint       <- int                  : assimilation interval
  # i_s        <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e        <- int
  # amp_b      <- float
  # bc         <- np.array[N_MODEL]       : boundary condition if needed
  # return     -> np.array[dimc]       : assimilated field

  if not (i_s == 0 and i_e == N_MODEL):
    raise Exception("fdvar_analytical_innerloop() is not for non-coupled. i_s = %d and i_e = %d is given." % (i_s, i_e))

  m = np.asmatrix(finite_time_tangent_using_nonlinear(fcst_0_nda, DT, aint))
  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])
  fcst_0 = np.asmatrix(fcst_0_nda).T

  d = yo - h * fcst_0
  mt_ht_ri = m.T * h.T * r.I
  delta_x0 = (b.I + mt_ht_ri * h * m).I * mt_ht_ri * d
  anl_0_nda = fcst_0_nda + delta_x0.A.flatten()

  anl_1_nda = np.copy(anl_0_nda)
  for i in range(aint):
    anl_1_nda = model.timestep(anl_1_nda, DT, i_s, i_e, bc)

  return anl_1_nda

def fdvar_2j(anl_0_nda, fcst_0_nda, h_nda, r_nda, yo_nda, aint, i_s, i_e, amp_b, bc):
  # anl_0_nda  <- np.array[dimc]       : temporary analysis field
  # fcst_0_nda <- np.array[dimc]       : first guess field
  # h_nda      <- np.array[pc_obs, dimc] : observation operator
  # r_nda      <- np.array[pc_obs, pc_obs] : observation error covariance
  # yo_nda     <- np.array[pc_obs, 1]    : observation
  # aint       <- int                  : assimilation interval
  # i_s        <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e        <- int
  # amp_b      <- float
  # bc         <- np.array[N_MODEL]       : boundary condition if needed
  # return     -> float                : cost function 2J

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])
  anl_0  = np.asmatrix(anl_0_nda).T
  fcst_0 = np.asmatrix(fcst_0_nda).T

  anl_1_nda = np.copy(anl_0_nda)
  for i in range(0, aint):
    anl_1_nda = model.timestep(anl_1_nda, DT, i_s, i_e, bc)

  # all array-like objects below are np.matrix
  anl_1 = np.matrix(anl_1_nda).T
  twoj = (anl_0 - fcst_0).T * b.I * (anl_0 - fcst_0) + \
         (h * anl_1 - yo).T * r.I * (h * anl_1 - yo)
  return twoj[0,0]

def fdvar_2j_deriv(anl_0_nda, fcst_0_nda, h_nda, r_nda, yo_nda, aint, i_s, i_e, amp_b, bc=None):
  # anl_0_nda  <- np.array[dimc]       : temporary analysis field
  # fcst_0_nda <- np.array[dimc]       : first guess field
  # h_nda      <- np.array[pc_obs, dimc] : observation operator
  # r_nda      <- np.array[pc_obs, pc_obs] : observation error covariance
  # yo_nda     <- np.array[pc_obs, 1]    : observation
  # aint       <- int                  : assimilation interval
  # i_s        <- int                  : model grid number, assimilate only [i_s, i_e)
  # i_e        <- int
  # amp_b      <- float
  # return     -> np.array[dimc]       : gradient of cost function 2J

  if i_s != 0 or i_e != N_MODEL:
    raise Exception("method fdvar_2j_deriv does not support non/weakly coupled DA")

  h  = np.asmatrix(h_nda)
  r  = np.asmatrix(r_nda)
  yo = np.asmatrix(yo_nda)
  b  = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])
  anl_0  = np.asmatrix(anl_0_nda).T
  fcst_0 = np.asmatrix(fcst_0_nda).T

  m = finite_time_tangent(fcst_0_nda, DT, aint)
  inc = anl_0 - fcst_0
  fcst_1_nda = np.copy(fcst_0_nda)
  for i in range(aint):
    fcst_1_nda = model.timestep(fcst_1_nda, DT)
  fcst_1 = np.asmatrix(fcst_1_nda).T
  d = yo - h * fcst_1

  j_deriv = b.I * inc + (m.T * h.T * r.I) * (h * m * inc - d)

  return j_deriv.A.flatten() * 2.0

