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
VRFS = int(STEPS * 0.85) # verification period: [VRFS,STEPS)
FCST_LT = 5

OERR = 5.0
FERR_INI = 10.0

EXPLIST = [ \
  # # {"name":"etkf_non_clim_int8",  "rho":1.1, "aint":8, "nmem":10, \
  # #       "method":"etkf", "couple":"none", "bc":"climatology"}, \
  # {"name":"etkf_non_int8",  "rho":1.1, "aint":8, "nmem":10, \
  #       "method":"etkf", "couple":"none", "bc":"persistent"}, \
  # {"name":"etkf_weak_int8",  "rho":1.1, "aint":8, "nmem":10, \
  #       "method":"etkf", "couple":"weak"}, \
  # {"name":"etkf_strong_int8",  "rho":1.1, "aint":8, "nmem":10, \
  #       "method":"etkf", "couple":"strong"}, \
  # # {"name":"tdvar_non_clim_int8", "aint":8, \
  # #       "nmem":1, "method":"3dvar", "couple":"none", "bc":"climatology"}, \
  # {"name":"tdvar_non_int8", "aint":8, "amp_b":0.6, \
  #       "nmem":1, "method":"3dvar", "couple":"none", "bc":"persistent"}, \
  # {"name":"tdvar_weak_int8", "aint":8, "amp_b":0.6, \
  #       "nmem":1, "method":"3dvar", "couple":"weak"}, \
  # {"name":"tdvar_strong_int8", "aint":8, "amp_b":0.6, \
  #       "nmem":1, "method":"3dvar", "couple":"strong"}, \
  # {"name":"fdvar_non_clim_int8", "aint":8, \
  #       "nmem":1, "method":"4dvar", "couple":"none", "bc":"climatology"}, \
  {"name":"fdvar_non_int8_b01", "aint":8, "amp_b":0.1, \
        "nmem":1, "method":"4dvar", "couple":"none", "bc":"persistent"}, \
  {"name":"fdvar_weak_int8_b01", "aint":8, "amp_b":0.1, \
        "nmem":1, "method":"4dvar", "couple":"weak"}, \
  {"name":"fdvar_strong_int8_b01", "aint":8, "amp_b":0.1, \
        "nmem":1, "method":"4dvar", "couple":"strong"}, \
  {"name":"fdvar_non_int8_b015", "aint":8, "amp_b":0.15, \
        "nmem":1, "method":"4dvar", "couple":"none", "bc":"persistent"}, \
  {"name":"fdvar_weak_int8_b015", "aint":8, "amp_b":0.15, \
        "nmem":1, "method":"4dvar", "couple":"weak"}, \
  {"name":"fdvar_strong_int8_b015", "aint":8, "amp_b":0.15, \
        "nmem":1, "method":"4dvar", "couple":"strong"}, \
  {"name":"fdvar_non_int8_b06", "aint":8, "amp_b":0.6, \
        "nmem":1, "method":"4dvar", "couple":"none", "bc":"persistent"}, \
  {"name":"fdvar_weak_int8_b06", "aint":8, "amp_b":0.6, \
        "nmem":1, "method":"4dvar", "couple":"weak"}, \
  {"name":"fdvar_strong_int8_b06", "aint":8, "amp_b":0.6, \
        "nmem":1, "method":"4dvar", "couple":"strong"}, \
  {"name":"fdvar_non_int8_b08", "aint":8, "amp_b":0.8, \
        "nmem":1, "method":"4dvar", "couple":"none", "bc":"persistent"}, \
  {"name":"fdvar_weak_int8_b08", "aint":8, "amp_b":0.8, \
        "nmem":1, "method":"4dvar", "couple":"weak"}, \
  {"name":"fdvar_strong_int8_b08", "aint":8, "amp_b":0.8, \
        "nmem":1, "method":"4dvar", "couple":"strong"} \
]

Calc_lv = False

def getr():
  r = np.identity(DIMO) * (OERR * OERR)
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

