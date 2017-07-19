#!/usr/bin/env python

import math
import numpy as np

# note: "dimc" in comments is dimension of component (for non/weakly coupled)
DIMM = 9    # dimension of model variable n
DIMO = DIMM # dimension of observation variable m

DT = 0.01
TMAX = 100
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 5

OERR_A = 1.0
OERR_O = 5.0
FERR_INI = 10.0
AINT = 8

rho = "adaptive"
nmem = 10
amp_b_tdvar = 2.0
amp_b_fdvar = 1.5

EXPLIST = [ \
  # {"name":"etkf_non_10mem", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"none", "bc":"persistent", "r_local":None}, \
  # {"name":"etkf_weak_10mem", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"weak", "r_local":None}, \
  {"name":"full", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"full"}, \
  {"name":"band", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"band"}, \
  {"name":"horizontal", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"horizontal"}, \
  {"name":"vertical", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"vertical"}, \
  {"name":"3-components", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"3-components"}, \
  {"name":"atmos_sees_ocean", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"atmos_sees_ocean"}, \
  {"name":"ocean_sees_atmos", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"ocean_sees_atmos"}, \
  # {"name":"horizontal", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"horizontal"}, \
  {"name":"trop_sees_ocean", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"trop_sees_ocean"}, \
  {"name":"ocean_sees_trop", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"ocean_sees_trop"}, \
  # {"name":"dynamical", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"dynamical"}, \
  # {"name":"correlation", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation"}, \
  # {"name":"covariance", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"covariance"}, \
  # {"name":"random", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"random"}, \
  # {"name":"tdvar_non_b2_ind", "amp_b":amp_b_tdvar, "nmem":1, "method":"3dvar", "couple":"none", "bc":"independent"}, \
  # {"name":"tdvar_non_b2", "amp_b":amp_b_tdvar, "nmem":1, "method":"3dvar", "couple":"none", "bc":"persistent"}, \
  # {"name":"tdvar_weak_b2", "amp_b":amp_b_tdvar, "nmem":1, "method":"3dvar", "couple":"weak"}, \
  # {"name":"tdvar_strong_b2", "amp_b":amp_b_tdvar, "nmem":1, "method":"3dvar", "couple":"strong"}, \
  # {"name":"fdvar_non_b15_ind", "amp_b":amp_b_fdvar, "nmem":1, "method":"4dvar", "couple":"none", "bc":"independent"}, \
  # {"name":"fdvar_non_b15", "amp_b":amp_b_fdvar, "nmem":1, "method":"4dvar", "couple":"none", "bc":"persistent"}, \
  # {"name":"fdvar_weak_b15", "amp_b":amp_b_fdvar, "nmem":1, "method":"4dvar", "couple":"weak"}, \
  # {"name":"fdvar_strong_b15", "amp_b":amp_b_fdvar, "nmem":1, "method":"4dvar", "couple":"strong"}, \
]

Calc_lv = False

def getr():
  r = np.identity(DIMO) * OERR_A ** 2
  if DIMM == 9:
    for i in range(6, 9):
      r[i,i] = OERR_O ** 2
  return r

def geth():
  # DIMO == DIMM is assumed
  h = np.zeros((DIMO,DIMM))
  for i in range(0, DIMM):
    h[i,i] = 1.0
  return h

def debug_obj_print(obj, scope):
  print([k for k, v in scope.items() if id(obj) == id(v)])
  print(type(obj))
  print(obj)
  print()
  return 0

