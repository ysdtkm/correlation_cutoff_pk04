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
EXPLIST = [ ["xyz_long", 1.20, 25,[1.0,1.0,1.0], 3, "etkf"], \
            ["xyz_short",1.05, 8, [1.0,1.0,1.0], 3, "etkf"], \
            ["xy",       1.05, 8, [1.0,1.0,0.0], 3, "etkf"], \
            ["yz",       1.05, 8, [0.0,1.0,1.0], 3, "etkf"], \
            ["zx",       1.05, 8, [1.0,0.0,1.0], 3, "etkf"], \
            ["x",        1.05, 8, [1.0,0.0,0.0], 3, "etkf"], \
            ["y",        1.05, 8, [0.0,1.0,0.0], 3, "etkf"], \
            ["z",        1.05, 8, [0.0,0.0,1.0], 3, "etkf"], \
            ["xyz_3dvar",1.0 , 8, [1.0,1.0,1.0], 1, "3dvar"],\
            ["xyz_4dvar",1.0 , 8, [1.0,1.0,1.0], 1, "4dvar"] ]

def getr():
  r = np.matrix(np.identity(DIMO)) * (OERR * OERR)
  return r

def geth(diag_h):
  # DIMO == DIMM is assumed
  h = np.matrix(np.zeros((DIMO,DIMM)))
  for i in range(0, DIMM):
    h[i,i] = diag_h[i]
  return h

