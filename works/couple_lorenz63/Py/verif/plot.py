#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from const import *

def plot(name, nmem):
  hist_true = np.fromfile("data/%s_true.bin" % name, np.float64)
  hist_true = hist_true.reshape((STEPS, DIMM))
  hist_fcst = np.fromfile("data/%s_fcst.bin" % name, np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))
  hist_obs = np.fromfile("data/%s_obs.bin" % name, np.float64)
  hist_obs = hist_obs.reshape((STEPS, DIMO))

  hist_fcst_mean = np.mean(hist_fcst, axis=1)
  hist_fcst_sprd = np.sqrt(np.mean(hist_fcst**2, axis=1) - hist_fcst_mean**2)
  hist_err = hist_fcst_mean - hist_true

  # RMSE and Spread time average
  rmse = np.sqrt(np.mean(1.0/DIMM*hist_err[VRFS:STEPS,:]**2, axis=(0,1)))
  if (nmem > 1):
    sprd = np.sqrt(nmem/(nmem - 1.0)/DIMM * \
      np.mean(hist_fcst_sprd[VRFS:STEPS,:]**2, axis=(0,1)))
  else:
    sprd = 0.0

  # RMSE-Spread time series
  plt.rcParams["font.size"] = 16
  plt.yscale('log')
  plt.plot(np.sqrt(np.mean(1.0/DIMM*hist_err**2, axis=1)), label="RMSE")
  if (nmem > 1):
    plt.plot(np.sqrt(nmem/(nmem - 1.0) * \
    np.mean(hist_fcst_sprd**2, axis=1)), label="Spread")
  plt.legend()
  plt.xlabel("timestep")
  plt.title("[%s] RMSE:%6g Spread:%6g" % (name, rmse, sprd))
  plt.savefig("./image/%s_%s.png" % (name, 1))
  plt.clf()

  # xyz time series
  plt.rcParams["font.size"] = 12
  fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
  ax1.set_title(name)
  ax1.plot(hist_true[:,0], label="true")
  ax1.plot(hist_fcst_mean[:,0], label="model")
  if "x" in name:
    ax1.plot(hist_obs[:,0], label="obs", linestyle='None', marker=".")
  ax1.set_ylabel("x")
  ax1.legend(loc="upper right")
  ax2.plot(hist_true[:,1], label="true")
  ax2.plot(hist_fcst_mean[:,1], label="model")
  if "y" in name:
    ax2.plot(hist_obs[:,1], label="obs", linestyle='None', marker=".")
  ax2.set_ylabel("y")
  ax3.plot(hist_true[:,2], label="true")
  ax3.plot(hist_fcst_mean[:,2], label="model")
  if "z" in name:
    ax3.plot(hist_obs[:,2], label="obs", linestyle='None', marker=".")
  ax3.set_ylabel("z")
  plt.xlabel("timestep")
  plt.savefig("./image/%s_%s.png" % (name, 2))
  plt.clf()

  # # 3D trajectory
  # plt.rcParams["font.size"] = 16
  # fig = plt.figure()
  # fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, \
  #   wspace=0.04, hspace=0.04)
  # ax = fig.add_subplot(111, projection='3d')
  # ax.plot(hist_true[:,0], hist_true[:,1], \
  #   hist_true[:,2], label="true")
  # ax.plot(hist_fcst_mean[:,0], hist_fcst_mean[:,1], \
  #   hist_fcst_mean[:,2], label="model")
  # ax.legend()
  # ax.set_xlim([-30,30])
  # ax.set_ylim([-30,30])
  # ax.set_zlim([0,50])
  # ax.set_xlabel("x")
  # ax.set_ylabel("y")
  # ax.set_zlabel("z")
  # plt.savefig("./image/%s_traj.png" % name)
  # plt.clf()
  # plt.close()

def methods(obj):
  print([method for method in dir(obj) if callable(getattr(obj, method))])

for exp in EXPLIST:
  plot(exp[0], exp[4])
