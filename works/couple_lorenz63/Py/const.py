#!/usr/bin/env python

import math
import numpy as np

DIMM = 40  # dimension of model variable n
DIMO = DIMM # dimension of observation variable m

DT = 0.01
TMAX = 120
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
VRFS = int(STEPS * 0.25) # verification period: [VRFS,STEPS)
FCST_LT = 5

OERR = 0.1
FERR_INI = 10.0

EXPLIST = [ \
  {"name":"etkf_no_infl",  "inf":1.0, "aint":10, \
        "diag":np.ones(DIMM), "nmem":DIMM+1, \
        "method":"etkf", "couple":"strong"}, \
  # {"name":"etkf_non_int25",  "inf":1.2, "aint":25, \
  #       "diag":np.ones(DIMM), "nmem":4, \
  #       "method":"etkf", "couple":"none"}, \
  # {"name":"etkf_weak_int25",  "inf":1.2, "aint":25, \
  #       "diag":np.ones(DIMM), "nmem":4, \
  #       "method":"etkf", "couple":"weak"}, \
  # {"name":"etkf_strong_int25",  "inf":1.2, "aint":25, \
  #       "diag":np.ones(DIMM), "nmem":4, \
  #       "method":"etkf", "couple":"strong"}, \
  # {"name":"tdvar_non_int25", "inf":1.0, "aint":25, \
  #       "diag":np.ones(DIMM), \
  #       "nmem":1, "method":"3dvar", "couple":"none"}, \
  # {"name":"tdvar_weak_int25", "inf":1.0, "aint":25, \
  #       "diag":np.ones(DIMM), \
  #       "nmem":1, "method":"3dvar", "couple":"weak"}, \
  # {"name":"tdvar_strong_int25", "inf":1.0, "aint":25, \
  #       "diag":np.ones(DIMM), \
  #       "nmem":1, "method":"3dvar", "couple":"strong"}, \
  # {"name":"fdvar_non_int25", "inf":1.0, "aint":25, \
  #       "diag":np.ones(DIMM), \
  #       "nmem":1, "method":"4dvar", "couple":"none"}, \
  # {"name":"fdvar_weak_int25", "inf":1.0, "aint":25, \
  #       "diag":np.ones(DIMM), \
  #       "nmem":1, "method":"4dvar", "couple":"weak"}, \
  # {"name":"fdvar_strong_int25", "inf":1.0, "aint":25, \
  #       "diag":np.ones(DIMM), \
  #       "nmem":1, "method":"4dvar", "couple":"strong"} \
]

def getr():
  r = np.identity(DIMO) * (OERR * OERR)
  return r

def geth(diag_h):
  # DIMO == DIMM is assumed
  h = np.zeros((DIMO,DIMM))
  for i in range(0, DIMM):
    h[i,i] = diag_h[i]
  return h

def debug_obj_print(obj, scope):
  print([k for k, v in scope.items() if id(obj) == id(v)])
  print(type(obj))
  print(obj)
  print()
  return 0

