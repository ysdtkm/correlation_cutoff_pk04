#!/usr/bin/env python

import numpy as np
from const import *

# x      <- np.array(DIMM)
# dt     <- float
# return -> np.array(DIMM)
def timestep(x, dt):
  # np.array x[DIMM]
  k1 = tendency(x)
  x2 = x + k1 * dt / 2.0
  k2 = tendency(x2)
  x3 = x + k2 * dt / 2.0
  k3 = tendency(x3)
  x4 = x + k3 * dt
  k4 = tendency(x4)
  x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
  return x

# a31p63-64
# x      <- np.array(DIMM)
# return -> np.array(DIMM)
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

# x      <- np.array(DIMM)       : state vector at the beginning
# dt     <- float                : infinitesimal time
# return -> np.matrix(DIMM,DIMM) : instantaneous tangent linear matrix M
def tangent_linear(x, dt):
  dx = np.matrix(np.zeros((DIMM,DIMM)))

  if (DIMM == 3):
    sigma = 10.0
    r = 28.0
    b = 8.0 / 3.0

    dx[0,0] = -sigma
    dx[0,1] = sigma
    dx[0,2] = 0.0

    dx[1,0] = -x[2] + r
    dx[1,1] = -1.0
    dx[1,2] = -x[0]

    dx[2,0] = x[1]
    dx[2,1] = x[0]
    dx[2,2] = -b

  elif (DIMM == 9):
    sigma = 10.0
    r     = 28.0
    b     = 8.0 / 3.0
    tau   = 0.1
    c     = 1.0
    cz    = 1.0
    ce    = 0.08
    s     = 1.0
    k1    = 10.0
    k2    = -11.0

    # extratropic atm
    dx[0,0] = -sigma
    dx[0,1] = sigma
    dx[0,3] = -ce * s

    dx[1,0] = -x[2] + r
    dx[1,1] = -1.0
    dx[1,2] = -x[0]
    dx[1,4] = ce * s

    dx[2,0] = x[1]
    dx[2,1] = x[0]
    dx[2,2] = -b

    # tropic atm
    dx[3,0] = -ce * s
    dx[3,3] = -sigma
    dx[3,4] = sigma
    dx[3,6] = -c * s

    dx[4,1] = ce * s
    dx[4,3] = -x[5] + r
    dx[4,4] = -1.0
    dx[4,5] = -x[3]
    dx[4,7] = c * s

    dx[5,3] = x[4]
    dx[5,4] = x[3]
    dx[5,5] = -b
    dx[5,8] = cz

    # tropic ocn
    dx[6,3] = - c
    dx[6,6] = tau * (-sigma)
    dx[6,7] = tau * sigma

    dx[7,4] = c
    dx[7,6] = tau * (-s * x[8] + r)
    dx[7,7] = -tau
    dx[7,8] = tau * (-s * x[6])

    dx[8,5] = -cz
    dx[8,6] = tau * (s * x[7])
    dx[8,7] = tau * (s * x[6])
    dx[8,8] = tau * (-b)

  m = np.matrix(np.identity(DIMM) + dx * dt)
  return m

### 掛け算の方向が正しいか要チェック
# x0     <- np.array(DIMM)       : state vector at t0
# iw     <- int                  : integration window (time in steps)
# return -> np.matrix(DIMM,DIMM) : finite time (t0 -> t0 + iw*DT) tangent linear matrix M
def finite_time_tangent(x0, iw):
  m_finite = np.matrix(np.identity(DIMM))
  x = np.copy(x0)
  for i in range(iw):
    m_inst = tangent_linear(x, DT)
    m_finite = m_inst * m_finite
    x = timestep(x, DT)
  return m_finite

# ### TLの転置に一致するような順番（TLを求めてから積分）にしている
# # x0     <- np.array(DIMM)       : state vector at t0
# # iw     <- int                  : integration window (time in steps)
# # return -> np.matrix(DIMM,DIMM) : finite time (t0 <- t0 + iw*DT) adjoint matrix M.T
# def finite_time_adjoint(x0, iw):
#   mt_finite = np.matrix(np.identity(DIMM))
#   x = np.copy(x0)
#   for i in range(iw):
#     m_inst = tangent_linear(x, DT)
#     mt_finite = mt_finite * m_inst.T
#     x = timestep(x, DT)
#   return mt_finite


