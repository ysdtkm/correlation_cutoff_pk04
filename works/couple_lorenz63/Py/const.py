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
nmem = 6
amp_b_tdvar = 2.0
amp_b_fdvar = 1.5

# EXPLIST = [{"name":"tdvar", "amp_b":amp_b_tdvar, "nmem":1, "method":"3dvar", "couple":"strong"}]
# EXPLIST = [{"name":"fdvar", "amp_b":amp_b_fdvar, "nmem":1, "method":"4dvar", "couple":"strong"}]
# EXPLIST = [{"name":name, "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":name} \
#     for name in ["full", "see_adjacent", "trop_ocean_couple", "extra_trop_couple", "individual", \
#         "atmos_sees_ocean", "ocean_sees_atmos", "trop_sees_ocean", "ocean_sees_trop", "dynamical", "rms_correlation", "rms_covariance"]]
EXPLIST = [
{"name":"full", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"full"},
{"name":"covariance-rms_81", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"covariance-rms", "num_yes":81},
{"name":"covariance-rms_60", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"covariance-rms", "num_yes":60},
{"name":"covariance-rms_37", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"covariance-rms", "num_yes":37},
{"name":"bhhtri_81", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"bhhtri", "num_yes":81},
{"name":"bhhtri_60", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"bhhtri", "num_yes":60},
{"name":"bhhtri_37", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"bhhtri", "num_yes":37},
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

