#!/usr/bin/env python

import math
import numpy as np

DIMM = 9    # dimension of model variable n
DIMO = DIMM # dimension of observation variable m

DT = 0.01
TMAX = 2
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
VRFS = int(STEPS * 0.25) # verification period: [VRFS,STEPS)
FCST_LT = 5

OERR = math.sqrt(5.0)
FERR_INI = 10.0

# name, inf, aint, diag(h), nmem, method
EXPLIST = [ \
  {"name":"etkf_weak_int25",  "inf":1.2, "aint":25, \
        "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "nmem":4, \
        "method":"etkf", "couple":"weak"}, \
  {"name":"etkf_strong_int25",  "inf":1.2, "aint":25, \
        "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "nmem":4, \
        "method":"etkf", "couple":"strong"}, \
  {"name":"tdvar_weak_int25", "inf":1.0, "aint":25, \
        "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], \
        "nmem":1, "method":"3dvar", "couple":"weak"}, \
  {"name":"tdvar_strong_int25", "inf":1.0, "aint":25, \
        "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], \
        "nmem":1, "method":"3dvar", "couple":"strong"}, \
  {"name":"fdvar_weak_int25", "inf":1.0, "aint":25, \
        "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], \
        "nmem":1, "method":"4dvar", "couple":"weak"}, \
  {"name":"fdvar_strong_int25", "inf":1.0, "aint":25, \
        "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], \
        "nmem":1, "method":"4dvar", "couple":"strong"} \
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

