#!/usr/bin/env python

import math
import numpy as np

DIMM = 9    # dimension of model variable n
DIMO = DIMM # dimension of observation variable m

DT = 0.01
TMAX = 10
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
VRFS = int(STEPS * 0.25) # verification period: [VRFS,STEPS)
FCST_LT = 5

OERR = math.sqrt(2.0)
FERR_INI = 10.0

# name, inf, aint, diag(h), nmem, method
EXPLIST = [ \
  {"name":"etkf_xyz_long",  "inf":1.05, "aint":25, "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "nmem":9, "method":"etkf"}, \
  # {"name":"xyz_short", "inf":1.05, "aint":8,  "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "nmem":3, "method":"etkf"}, \
  # {"name":"xy",        "inf":1.05, "aint":8,  "diag":[1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0], "nmem":3, "method":"etkf"}, \
  # {"name":"yz",        "inf":1.05, "aint":8,  "diag":[0.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0], "nmem":3, "method":"etkf"}, \
  # {"name":"zx",        "inf":1.05, "aint":8,  "diag":[1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0], "nmem":3, "method":"etkf"}, \
  # {"name":"x",         "inf":1.05, "aint":8,  "diag":[1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0], "nmem":3, "method":"etkf"}, \
  # {"name":"y",         "inf":1.05, "aint":8,  "diag":[0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0], "nmem":3, "method":"etkf"}, \
  # {"name":"z",         "inf":1.05, "aint":8,  "diag":[0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0], "nmem":3, "method":"etkf"}, \
  # {"name":"xyz_3dvar", "inf":1.0 , "aint":8,  "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "nmem":1, "method":"3dvar"},\
  {"name":"fdvar_xyz_long", "inf":1.0 , "aint":25, "diag":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "nmem":1, "method":"4dvar"} \
]

def getr():
  r = np.matrix(np.identity(DIMO)) * (OERR * OERR)
  return r

def geth(diag_h):
  # DIMO == DIMM is assumed
  h = np.matrix(np.zeros((DIMO,DIMM)))
  for i in range(0, DIMM):
    h[i,i] = diag_h[i]
  return h

