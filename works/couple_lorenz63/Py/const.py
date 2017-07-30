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
nmem = 4
amp_b_tdvar = 2.0
amp_b_fdvar = 1.5

EXPLIST = [ \
  {"name":"correlation-clim_9", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":9},
  {"name":"correlation-clim_11", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":11},
  {"name":"correlation-clim_13", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":13},
  {"name":"correlation-clim_15", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":15},
  {"name":"correlation-clim_17", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":17},
  {"name":"correlation-clim_19", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":19},
  {"name":"correlation-clim_21", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":21},
  {"name":"correlation-clim_23", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":23},
  {"name":"correlation-clim_25", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":25},
  {"name":"correlation-clim_27", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":27},
  {"name":"correlation-clim_29", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":29},
  {"name":"correlation-clim_31", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":31},
  {"name":"correlation-clim_33", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":33},
  {"name":"correlation-clim_35", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":35},
  {"name":"correlation-clim_37", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":37},
  {"name":"correlation-clim_39", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":39},
  {"name":"correlation-clim_41", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":41},
  {"name":"correlation-clim_43", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":43},
  {"name":"correlation-clim_45", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":45},
  {"name":"correlation-clim_47", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":47},
  {"name":"correlation-clim_49", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":49},
  {"name":"correlation-clim_51", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":51},
  {"name":"correlation-clim_53", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":53},
  {"name":"correlation-clim_55", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":55},
  {"name":"correlation-clim_57", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":57},
  {"name":"correlation-clim_59", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":59},
  {"name":"correlation-clim_61", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":61},
  {"name":"correlation-clim_63", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":63},
  {"name":"correlation-clim_65", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":65},
  {"name":"correlation-clim_67", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":67},
  {"name":"correlation-clim_69", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":69},
  {"name":"correlation-clim_71", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":71},
  {"name":"correlation-clim_73", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":73},
  {"name":"correlation-clim_75", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":75},
  {"name":"correlation-clim_77", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":77},
  {"name":"correlation-clim_79", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":79},
  {"name":"correlation-clim_81", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"correlation-clim", "num_yes":81},
]

# {"name":"etkf", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"full"},
# {"name":"tdvar", "rho":rho, "nmem":1, "method":"3dvar", "couple":"strong", "amp_b":amp_b_tdvar},
# {"name":"fdvar", "rho":rho, "nmem":1, "method":"4dvar", "couple":"strong", "amp_b":amp_b_fdvar},

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

