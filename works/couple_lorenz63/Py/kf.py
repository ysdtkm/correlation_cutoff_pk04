#!/usr/bin/env python

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from numpy.linalg import inv, sqrtm

# ===============================
DIM = 3
DT = 0.01
TMAX = 3
STEPS = int(TMAX / DT)
OERR = math.sqrt(2.0)
FERR = 30.0
INF = 1.25
AINT = 8
method = "KF"
MEM = 3
# ===============================

def main():
  # for validation
  hist_true = np.empty((STEPS, DIM))
  hist_fcst = np.empty((STEPS, DIM))
  hist_kg = np.empty(STEPS)

  # initialization with free run
  true = np.random.normal(0.0, 50.0, DIM) # true
  fcst = np.random.normal(0.0, 50.0, DIM) # anl / fcst
  for i in range(0, int(50 / DT)):
    true = timestep(true)
    fcst = timestep(fcst)

  # initial parameters for analysis
  r = np.matrix(np.identity(DIM) * (OERR * OERR)) # obs error covariance matrix (DIM, DIM)
  pa = np.matrix(np.identity(DIM)) * (FERR * FERR) # anl error covariance matrix (DIM, DIM)
  h = geth() # observation operator
  m = getm(fcst) # tangent linear model (DIM, DIM)

  # analysis cycle
  for i in range(0, STEPS):
    true = timestep(true)
    fcst = timestep(fcst)
    hist_true[i, :] = true
    hist_fcst[i, :] = fcst

    # assimilation (np.array -> np.matrix)
    if i % AINT == 0:
      yo = np.matrix(true + np.random.normal(0.0, OERR, DIM)).T
      xf = np.matrix(fcst).T
      iden = np.matrix(np.identity(DIM))

      if (method == "KF"):
        pf = m * pa * (m.T) * (INF * INF)
        kg = pf * h.T * inv(h * pf * h.T + r)
        xa = xf + kg * (yo - h * xf)
        pa = (iden - kg * h) * pf
      elif (method == "ETKF"):
        pf = m * pa * (m.T)
        ybm = h * pf
        yb = mean(ybm, 2) # ttk
        pa = inv(((MEM - 1)/INF**2) * iden + ybm.T * inv(r) * ybm)
        wam = sqrtm((MEM - 1) * pa)
        wa = pa * yb.T * inv(r) * (yo - yb)
        xap = xbp * wam
        xa = xf * xbm * wa # ttk
        xam = xap + xa * np.matrix(np.ones(MEM))


      fcst = np.array(xa.T[0,:]).flatten()
      m = getm(fcst)

    hist_kg[i] = np.mean(np.array(kg) ** 2)

  # plot
  plot(hist_kg, hist_true, hist_fcst)

def timestep(x):
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

  k = np.empty((DIM))
  k[0] = -sigma * x[0]            + sigma * x[1]
  k[1] =  -x[0] * x[2] + r * x[0] -         x[1]
  k[2] =   x[0] * x[1] - b * x[2]
  return k

def geth(): # observation matrix
  h = np.matrix(np.identity(DIM))
  h[0,0] = 0
  h[2,2] = 0
  return h

def getm(x): # tangent linear model (DIM, DIM)
  m = np.matrix(np.empty((DIM, DIM)))
  alpha = 1.0e-4
  for i in range(0, DIM):
    dx = np.zeros(DIM)
    dx[i] = alpha
    x_ctl = x
    x_ptb = x + dx
    for j in range(0, AINT):
      x_ctl = timestep(x_ctl)
      x_ptb = timestep(x_ptb)
    m[:, i] = (x_ptb - x_ctl)[:,np.newaxis] / alpha
  return m

def plot(hist_kg, hist_true, hist_fcst):
  hist_err = hist_fcst - hist_true
  draw(hist_kg, 1)
  draw(hist_err, 2)
  draw(hist_true[:,0], 3)
  draw(hist_fcst[:,0], 4)
  # draw(np.array(pf), 3)
  # draw(np.mean(hist_err ** 2, 1) ** 0.5, 3)
  # draw(np.mean(np.absolute(hist_err[int(STEPS / 2) : STEPS, :]), 0), 4)
  # draw(np.absolute(hist_err))

def draw(data, name):
  if len(data.shape) == 2:
    plt.pcolor(data, cmap=matplotlib.pyplot.cm.bwr)
    val = np.max(np.absolute(data))
    plt.clim((-1 * val), val)
    plt.colorbar()
  else:
    if data.min() > 0:
      plt.yscale('log')
    plt.plot(data)
  plt.savefig("./image/%s.png" % name)
  plt.clf()

main()

os.system("cd latex; make")
