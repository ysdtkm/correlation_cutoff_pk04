#!/usr/bin/env python

import numpy as np
from scipy.optimize import fmin_bfgs
import sys, os
py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_dir)
from control.const import *

def tdvar(fcst, h, r, yo):
  anl = fcst
  anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo))
  return anl.T

def tdvar_2j(anl, fcst, h, r, yo):
  b = 1.2 * tdvar_b()
  anl_tmp = anl
  anl = np.matrix(anl_tmp).T
  twoj = (anl - fcst).T * np.linalg.inv(b) * (anl - fcst) + \
       (h * anl - yo).T * np.linalg.inv(r) * (h * anl - yo)
  return twoj

def tdvar_b():
  # obtained from off-line ETKF (500-1999th step mean(xbpt*xbpt.T))
  # b = np.matrix([ \
  #   [ 0.34897187,0.5587116,-0.08293288], \
  #   [ 0.5587116, 1.10975806,0.00229167], \
  #   [-0.08293288,0.00229167,0.60078791]  \
  # ])
  b = np.matrix([ \
    [1,0,0,0,0,0,0,0,0], \
    [0,1,0,0,0,0,0,0,0], \
    [0,0,1,0,0,0,0,0,0], \
    [0,0,0,1,0,0,0,0,0], \
    [0,0,0,0,1,0,0,0,0], \
    [0,0,0,0,0,1,0,0,0], \
    [0,0,0,0,0,0,1,0,0], \
    [0,0,0,0,0,0,0,1,0], \
    [0,0,0,0,0,0,0,0,1] \
  ])
  b = np.matrix([ \
    [1,0,0], \
    [0,1,0], \
    [0,0,1] \
  ])
  return b

