#!/usr/bin/env python

import math

DIMM = 3    # dimension of model variable n
DIMO = DIMM # dimension of observation variable m

DT = 0.01
TMAX = 20
STEPS = int(TMAX / DT)
VRFS = int(STEPS * 0.25) # verification period: [VRFS,STEPS)

OERR = math.sqrt(2.0)
FERR_INI = 10.0

# name, inf, aint, diag(h), nmem, method
EXPLIST = [ \
  {"name":"xyz_long",  "inf":1.20, "aint":25, "diag":[1.0,1.0,1.0], "nmem":3, "method":"etkf"}, \
  {"name":"xyz_short", "inf":1.05, "aint":8,  "diag":[1.0,1.0,1.0], "nmem":3, "method":"etkf"}, \
  {"name":"xy",        "inf":1.05, "aint":8,  "diag":[1.0,1.0,0.0], "nmem":3, "method":"etkf"}, \
  {"name":"yz",        "inf":1.05, "aint":8,  "diag":[0.0,1.0,1.0], "nmem":3, "method":"etkf"}, \
  {"name":"zx",        "inf":1.05, "aint":8,  "diag":[1.0,0.0,1.0], "nmem":3, "method":"etkf"}, \
  {"name":"x",         "inf":1.05, "aint":8,  "diag":[1.0,0.0,0.0], "nmem":3, "method":"etkf"}, \
  {"name":"y",         "inf":1.05, "aint":8,  "diag":[0.0,1.0,0.0], "nmem":3, "method":"etkf"}, \
  {"name":"z",         "inf":1.05, "aint":8,  "diag":[0.0,0.0,1.0], "nmem":3, "method":"etkf"}, \
  {"name":"xyz_3dvar", "inf":1.0 , "aint":8,  "diag":[1.0,1.0,1.0], "nmem":1, "method":"3dvar"},\
  {"name":"xyz_4dvar", "inf":1.0 , "aint":8,  "diag":[1.0,1.0,1.0], "nmem":1, "method":"4dvar"} \
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
