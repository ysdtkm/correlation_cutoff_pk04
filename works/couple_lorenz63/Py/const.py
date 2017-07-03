#!/usr/bin/env python

import math
import numpy as np

# note: "dimc" in comments is dimension of component (for non/weakly coupled)
DIMM = 9    # dimension of model variable n
DIMO = DIMM # dimension of observation variable m

DT = 0.01
TMAX = 10
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 5

OERR_A = 1.0
OERR_O = 5.0
FERR_INI = 10.0
AINT = 8

EXPLIST = [ \
  {"name":"etkf_non_2mem", "rho":1.1, "nmem":2, "method":"etkf", "couple":"none", "bc":"persistent"}, \
  {"name":"etkf_weak_2mem", "rho":1.1, "nmem":2, "method":"etkf", "couple":"weak"}, \
  {"name":"etkf_strong_2mem", "rho":1.1, "nmem":2, "method":"etkf", "couple":"strong"}, \
  {"name":"etkf_non_3mem", "rho":1.1, "nmem":3, "method":"etkf", "couple":"none", "bc":"persistent"}, \
  {"name":"etkf_weak_3mem", "rho":1.1, "nmem":3, "method":"etkf", "couple":"weak"}, \
  {"name":"etkf_strong_3mem", "rho":1.1, "nmem":3, "method":"etkf", "couple":"strong"}, \
  {"name":"etkf_non_4mem", "rho":1.1, "nmem":4, "method":"etkf", "couple":"none", "bc":"persistent"}, \
  {"name":"etkf_weak_4mem", "rho":1.1, "nmem":4, "method":"etkf", "couple":"weak"}, \
  {"name":"etkf_strong_4mem", "rho":1.1, "nmem":4, "method":"etkf", "couple":"strong"}, \
  {"name":"etkf_non_6mem", "rho":1.1, "nmem":6, "method":"etkf", "couple":"none", "bc":"persistent"}, \
  {"name":"etkf_weak_6mem", "rho":1.1, "nmem":6, "method":"etkf", "couple":"weak"}, \
  {"name":"etkf_strong_6mem", "rho":1.1, "nmem":6, "method":"etkf", "couple":"strong"}, \
  {"name":"etkf_non_10mem", "rho":1.1, "nmem":10, "method":"etkf", "couple":"none", "bc":"persistent"}, \
  {"name":"etkf_weak_10mem", "rho":1.1, "nmem":10, "method":"etkf", "couple":"weak"}, \
  {"name":"etkf_strong_10mem", "rho":1.1, "nmem":10, "method":"etkf", "couple":"strong"} \
  # {"name":"tdvar_non_b5", "amp_b":5.0, "nmem":1, "method":"3dvar", "couple":"none", "bc":"persistent"}, \
  # {"name":"tdvar_weak_b5", "amp_b":5.0, "nmem":1, "method":"3dvar", "couple":"weak"}, \
  # {"name":"tdvar_strong_b5", "amp_b":5.0, "nmem":1, "method":"3dvar", "couple":"strong"}, \
  # {"name":"fdvar_non_b3", "amp_b":3.0, "nmem":1, "method":"4dvar", "couple":"none", "bc":"persistent"}, \
  # {"name":"fdvar_weak_b3", "amp_b":3.0, "nmem":1, "method":"4dvar", "couple":"weak"}, \
  # {"name":"fdvar_strong_b3", "amp_b":3.0, "nmem":1, "method":"4dvar", "couple":"strong"} \
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

