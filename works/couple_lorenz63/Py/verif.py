#!/usr/bin/env python

import numpy as np
from const import *

def verif():
  hist_true = np.fromfile("data/true.bin", np.float64)
  hist_true = hist_true.reshape((NEXP, STEPS, DIMM))
  hist_fcst = np.fromfile("data/fcst.bin", np.float64)
  hist_fcst = hist_fcst.reshape((NEXP, STEPS, NMEM, DIMM))

# verify observation minus background (yo - yb) at each DA step
def obs_minus_back():

# verify analysis minus true at each DA step
def anl_minus_true():

# verify background minus true at each DA step
def back_minus_true():

# verify backgromnd RMSE (from true) vs spread
def rmse_vs_sprd():

verif()
