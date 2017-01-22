#!/usr/bin/env python

import numpy as np
from const import *

def timestep(x):
  # np.array x[DIMM]
  k1 = tendency(x)
  x2 = x + k1 * DT / 2.0
  k2 = tendency(x2)
  x3 = x + k2 * DT / 2.0
  k3 = tendency(x3)
  x4 = x + k3 * DT
  k4 = tendency(x4)
  x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * DT / 6.0
  return x

def tendency(x):
  sigma = 10.0
  r = 28.0
  b = 8.0 / 3.0

  k = np.empty((DIMM))
  k[0] = -sigma * x[0]            + sigma * x[1]
  k[1] =  -x[0] * x[2] + r * x[0] -         x[1]
  k[2] =   x[0] * x[1] - b * x[2]
  return k

