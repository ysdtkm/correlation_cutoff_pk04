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
FCST_LT = 0

OERR_A = 1.0
OERR_O = 5.0
FERR_INI = 10.0
AINT = 8

rho = "adaptive"
nmem = <<param1>>
amp_b_tdvar = 2.0
amp_b_fdvar = 1.5

EXPLIST = [ \
  {"name":"<<param2>>_<<param3_sanit>>", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"<<param2>>", "num_yes":<<param3>>},
]

# {"name":"etkf", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"full"},
# {"name":"tdvar", "rho":rho, "nmem":1, "method":"3dvar", "couple":"strong", "amp_b":amp_b_tdvar},
# {"name":"fdvar", "rho":rho, "nmem":1, "method":"4dvar", "couple":"strong", "amp_b":amp_b_fdvar},

Calc_lv = False

ETKF_vo = 1.0
ETKF_kappa = 1.01
ETKF_AI_max = 1.2
ETKF_AI_min = 0.9

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

