#!/usr/bin/env python

import numpy as np
from const import *

# input:  np.array(DIMM)
# output: np.array(DIMM)
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

# a31p63-64
# input:  np.array(DIMM)
# output: np.array(DIMM)
def tendency(x):
  if (DIMM == 3):
    sigma = 10.0
    r = 28.0
    b = 8.0 / 3.0
    k = np.empty((3))
    k[0] = -sigma * x[0]            + sigma * x[1]
    k[1] =  -x[0] * x[2] + r * x[0] -         x[1]
    k[2] =   x[0] * x[1] - b * x[2]
    return k

  elif (DIMM == 9):
    ### model constants
    # dynamic
    sigma = 10.0
    r     = 28.0
    b     = 8.0 / 3.0
    # coupling
    tau   = 0.1
    c     = 1.0
    cz    = 1.0
    ce    = 0.08
    # offset and scaling
    s     = 1.0
    k1    = 10.0
    k2    = -11.0

    dx = np.empty((DIMM))
    # extratropic atm
    dx[0] =           -sigma * x[0] + sigma * x[1] - ce * (s * x[3] + k1)
    dx[1] = -x[0] * x[2] + r * x[0] -         x[1] + ce * (s * x[4] + k1)
    dx[2] =  x[0] * x[1] - b * x[2]
    # tropic atm
    dx[3] =           -sigma * x[3] + sigma * x[4] - c  * (s * x[6] + k2) - ce * (s * x[0] + k1)
    dx[4] = -x[3] * x[5] + r * x[3] -         x[4] + c  * (s * x[7] + k2) + ce * (s * x[1] + k1)
    dx[5] =  x[3] * x[4] - b * x[5]                + cz * x[8]
    # tropic ocn
    dx[6] = tau * (              -sigma * x[6] + sigma * x[7]) - c  * (x[3] + k2)
    dx[7] = tau * (-s * x[6] * x[8] + r * x[6] -         x[7]) + c  * (x[4] + k2)
    dx[8] = tau * ( s * x[6] * x[7] - b * x[8]               ) - cz * x[5]
    return dx

def tangent_linear(x):
  if (DIMM == 3):
    sigma = 10.0
    r = 28.0
    b = 8.0 / 3.0
    k = np.zeros((3,3))

    k[0][0] = -sigma
    k[0][1] = sigma
    k[0][2] = 0.0

    k[1][0] = -x[2] + r
    k[1][1] = -1.0
    k[1][2] = -x[0]

    k[2][0] = x[1]
    k[2][1] = x[0]
    k[2][2] = -b
    return k

