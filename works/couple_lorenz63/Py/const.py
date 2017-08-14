#!/usr/bin/env python

import math
import numpy as np

# note: "dimc" in comments is dimension of component (for non/weakly coupled)
N_MODEL = 9    # dimension of model variable n
P_OBS = N_MODEL # dimension of observation variable m

DT = 0.01
TMAX = 2
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR_A = 1.0
OERR_O = 5.0
FERR_INI = 10.0
AINT = 8

rho = 1.05 # "adaptive"
nmem = 6
amp_b_tdvar = 2.0
amp_b_fdvar = 1.5

EXPLIST = [ \
  {"name":"etkf_strong", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"full"},
  {"name":"tdvar_strong", "nmem":1, "method":"3dvar", "couple":"strong", "amp_b":amp_b_tdvar},
  {"name":"fdvar_strong", "nmem":1, "method":"4dvar", "couple":"strong", "amp_b":amp_b_fdvar},
  {"name":"etkf_weak", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"weak", "r_local":"full"},
  {"name":"tdvar_weak", "nmem":1, "method":"3dvar", "couple":"weak", "amp_b":amp_b_tdvar},
  {"name":"fdvar_weak", "nmem":1, "method":"4dvar", "couple":"weak", "amp_b":amp_b_fdvar},
  {"name":"etkf_non", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"none", "r_local":"full", "bc":"persistence"},
  {"name":"tdvar_non", "nmem":1, "method":"3dvar", "couple":"none", "amp_b":amp_b_tdvar, "bc":"persistence"},
  {"name":"fdvar_non", "nmem":1, "method":"4dvar", "couple":"none", "amp_b":amp_b_fdvar, "bc":"persistence"},
]

# {"name":"etkf", "rho":rho, "nmem":nmem, "method":"etkf", "couple":"strong", "r_local":"full"},
# {"name":"tdvar", "nmem":1, "method":"3dvar", "couple":"strong", "amp_b":amp_b_tdvar},
# {"name":"fdvar", "nmem":1, "method":"4dvar", "couple":"strong", "amp_b":amp_b_fdvar},

Calc_lv = False

ETKF_vo = 1.0
ETKF_kappa = 1.01
ETKF_AI_max = 1.2
ETKF_AI_min = 0.9

def getr():
  # note: Non-diagonal element in R is ignored in exec_obs()
  r = np.identity(P_OBS) * OERR_A ** 2
  if N_MODEL == 9:
    if P_OBS == N_MODEL:
      for i in range(6, 9):
        r[i,i] = OERR_O ** 2
    else:
      import warnings
      warnings.warn("getr() ignores OERR_O if P_OBS != N_MODEL. P_OBS=%d, N_MODEL=%d was passed." % (P_OBS, N_MODEL))
  return r

def geth():
  # ttk h = np.zeros((P_OBS,N_MODEL))
  # for i in range(0, min(N_MODEL, P_OBS)):
  #   h[i,i] = 1.0
  # if P_OBS != N_MODEL:
  #   import warnings
  #   warnings.warn("geth() cannot correctly deal with P_OBS != N_MODEL. P_OBS=%d, N_MODEL=%d was passed." % (P_OBS, N_MODEL))
  h = np.diag([0, 1, 0, 0, 1, 0, 0, 1, 0])
  return h

def debug_obj_print(obj, scope):
  print([k for k, v in scope.items() if id(obj) == id(v)])
  print(type(obj))
  print(obj)
  print()
  return 0

