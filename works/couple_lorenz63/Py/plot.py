#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from const import *

## refer to a32p23
# name <- string
# nmem <- int
def plot_rmse_spread(name, nmem):
  hist_true = np.fromfile("data/true.bin", np.float64)
  hist_true = hist_true.reshape((STEPS, DIMM))
  hist_fcst = np.fromfile("data/%s_cycle.bin" % name, np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))

  # Error and Spread_square for each grid and time
  hist_fcst_mean = np.mean(hist_fcst, axis=1)
  hist_fcst_sprd2 = np.zeros((STEPS, DIMM))
  if (nmem > 1):
    for i in range(nmem):
      hist_fcst_sprd2[:,:] = hist_fcst_sprd2[:,:] + \
        1.0 / (nmem - 1.0) * (hist_fcst[:,i,:]**2 - hist_fcst_mean[:,:]**2)
  hist_err = hist_fcst_mean - hist_true

  for i_component in range(DIMM//3):
    grid_from = 3 * i_component
    name_component = ["extro", "trop", "ocn"][i_component]

    # MSE and Spread_square time series (grid average)
    mse_time   = np.mean(hist_err[:,grid_from:grid_from+3]**2,     axis=1)
    sprd2_time = np.mean(hist_fcst_sprd2[:,grid_from:grid_from+3], axis=1)

    # RMSE and Spread time average
    rmse = np.sqrt(np.mean(mse_time[STEP_FREE:STEPS]))
    sprd = np.sqrt(np.mean(sprd2_time[STEP_FREE:STEPS]))

    # RMSE-Spread time series
    plt.rcParams["font.size"] = 16
    plt.yscale('log')
    plt.plot(np.sqrt(mse_time), label="RMSE")
    if (nmem > 1):
      plt.plot(np.sqrt(sprd2_time), label="Spread")
    plt.legend()
    plt.xlabel("timestep")
    plt.title("[%s %s] RMSE:%6g Spread:%6g" % (name, name_component, rmse, sprd))
    plt.savefig("./image/%s_%s_%s.png" % (name, name_component, "time"))
    plt.clf()
  return 0

# name <- string
# nmem <- int
def plot_time_value(name, nmem):
  hist_true = np.fromfile("data/true.bin", np.float64)
  hist_true = hist_true.reshape((STEPS, DIMM))
  hist_fcst = np.fromfile("data/%s_cycle.bin" % name, np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))
  hist_obs = np.fromfile("data/%s_obs.bin" % name, np.float64)
  hist_obs = hist_obs.reshape((STEPS, DIMO))
  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  for i_component in range(DIMM//3):
    i_adjust = i_component * 3
    name_component = ["extro", "trop", "ocn"][i_component]

    # xyz time series
    plt.rcParams["font.size"] = 12
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.set_title("%s %s" % (name, name_component))
    ax1.plot(hist_true[:,0+i_adjust], label="true")
    ax1.plot(hist_fcst_mean[:,0+i_adjust], label="model")
    if "x" in name:
      ax1.plot(hist_obs[:,0+i_adjust], label="obs", linestyle='None', marker=".")
    ax1.set_ylabel("x")
    ax1.legend(loc="upper right")
    ax2.plot(hist_true[:,1+i_adjust], label="true")
    ax2.plot(hist_fcst_mean[:,1+i_adjust], label="model")
    if "y" in name:
      ax2.plot(hist_obs[:,1+i_adjust], label="obs", linestyle='None', marker=".")
    ax2.set_ylabel("y")
    ax3.plot(hist_true[:,2+i_adjust], label="true")
    ax3.plot(hist_fcst_mean[:,2+i_adjust], label="model")
    if "z" in name:
      ax3.plot(hist_obs[:,2+i_adjust], label="obs", linestyle='None', marker=".")
    ax3.set_ylabel("z")
    plt.xlabel("timestep")
    plt.savefig("./image/%s_%s_%s.png" % (name, name_component, "val"))
    plt.clf()
  return 0

# name <- string
# nmem <- int
def plot_3d_trajectory(name, nmem):
  hist_true = np.fromfile("data/true.bin", np.float64)
  hist_true = hist_true.reshape((STEPS, DIMM))
  hist_fcst = np.fromfile("data/%s_cycle.bin" % name, np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))
  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  for i_component in range(DIMM//3):
    i_adjust = i_component * 3
    name_component = ["extro", "trop", "ocn"][i_component]

    # 3D trajectory
    plt.rcParams["font.size"] = 16
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, \
      wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(hist_true[:,0+i_adjust], hist_true[:,1+i_adjust], \
      hist_true[:,2+i_adjust], label="true")
    ax.plot(hist_fcst_mean[:,0+i_adjust], hist_fcst_mean[:,1+i_adjust], \
      hist_fcst_mean[:,2+i_adjust], label="model")
    ax.legend()
    # ax.set_xlim([-30,30])
    # ax.set_ylim([-30,30])
    # ax.set_zlim([0,50])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("./image/%s_%s_traj.png" % (name, name_component))
    plt.clf()
    plt.close()
  return 0

# name <- string
def plot_covariance_matr(name):
  hist_covar = np.fromfile("data/%s_covr_post.bin" % name, np.float64)
  hist_covar = hist_covar.reshape((STEPS, DIMM, DIMM))
  mean_covar = np.sqrt(np.nanmean(hist_covar**2, axis=0))

  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(mean_covar))
  map1 = ax.pcolor(mean_covar, cmap=plt.cm.bwr)
  map1.set_clim(-1.0 * cmax, cmax)
  x0,x1 = ax.get_xlim()
  y0,y1 = ax.get_ylim()
  ax.set_aspect(abs(x1-x0)/abs(y1-y0))
  ax.set_xlabel("x")
  cbar = plt.colorbar(map1)
  plt.title("covariance matrix %s" % (name,))
  plt.gca().invert_yaxis()
  plt.savefig("./image/%s_covar.png" % (name,))
  plt.close()
  return 0

for exp in EXPLIST:
  plot_rmse_spread(exp["name"], exp["nmem"])
  plot_time_value(exp["name"], exp["nmem"])
  plot_3d_trajectory(exp["name"], exp["nmem"])
  if (exp["method"] == "etkf"):
    plot_covariance_matr(exp["name"])
