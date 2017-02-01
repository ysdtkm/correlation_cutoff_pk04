#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin_bfgs
import sys, os
py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_dir)
from control.const import *
from assim.tdvar import *
from model.model import *

def fdvar(fcst_0, h, r, yo, aint):
  # only assimilate one set of obs at t1 = t0+dt*aint
  # input fcst_0 is [aint] steps former than analysis time
  anl_0 = fcst_0
  anl_0 = fmin_bfgs(fdvar_2j, anl_0, args=(fcst_0, h, r, yo, aint))
  anl_1 = anl_0
  for i in range(0, aint):
    anl_1 = timestep(anl_1)
  return anl_1.T

def fdvar_2j(anl_0, fcst_0, h, r, yo, aint):
  b = 0.6 * 1.2 * tdvar_b()
  anl_tmp = anl_0
  anl_0 = np.matrix(anl_tmp).T
  anl_1_ar = anl_0.A.flatten()
  for i in range(0, aint):
    anl_1_ar = timestep(anl_1_ar)
  anl_1 = np.matrix(anl_1_ar).T
  twoj = (anl_0 - fcst_0).T * np.linalg.inv(b) * (anl_0 - fcst_0) + \
       (h * anl_1 - yo).T * np.linalg.inv(r) * (h * anl_1 - yo)
  return twoj.A

