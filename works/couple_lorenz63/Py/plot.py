#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os, warnings
from mpl_toolkits.mplot3d import Axes3D
from const import *

def plot_all():
  hist_true = np.fromfile("data/true.bin", np.float64)
  hist_true = hist_true.reshape((STEPS, DIMM))
  vectors = ["blv", "flv", "clv", "fsv", "isv"]
  vector_name = {"blv": "Backward_LV", "flv": "Forward_LV", "clv": "Characteristic_LV", \
                 "fsv": "Final_SV", "isv": "Initial_SV"}
  hist_vector = {}
  for vec in vectors:
    hist_vector[vec] = np.fromfile("data/%s.bin" % vec, np.float64)
    hist_vector[vec] = hist_vector[vec].reshape((STEPS, DIMM, DIMM))

  os.system("mkdir -p image/true")

  for vec in vectors:
    plot_lv_time(hist_vector[vec], vector_name[vec])
    # plot_trajectory_lv(hist_true, hist_vector[vec], vector_name[vec])

  for exp in EXPLIST:
    name = exp["name"]
    nmem = exp["nmem"]
    os.system("mkdir -p image/%s" % name)

    hist_fcst = np.fromfile("data/%s_cycle.bin" % name, np.float64)
    hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))
    hist_obs = np.fromfile("data/%s_obs.bin" % name, np.float64)
    hist_obs = hist_obs.reshape((STEPS, DIMO))

    plot_rmse_spread(hist_true, hist_fcst, name, nmem)
    plot_time_value(hist_true, hist_fcst, hist_obs, name, nmem)
    plot_3d_trajectory(hist_true, hist_fcst, name, nmem)

    if (exp["method"] == "etkf"):
      for vec in vectors:
        plot_lv_projection(hist_vector[vec], hist_fcst, name, vector_name[vec], nmem)

      for sel in ["back", "anl"]:
        hist_covar = np.fromfile("data/%s_covr_%s.bin" % (name, sel), np.float64)
        hist_covar = hist_covar.reshape((STEPS, DIMM, DIMM))
        plot_covariance_matr(hist_covar, name, sel)

def plot_lv_time(hist_lv, name):
  plt.rcParams["font.size"] = 12
  fig, ax1 = plt.subplots(1)
  ax1.set_title("lv1_1")
  ax1.plot(hist_lv[:,0,0], label="1")
  ax1.plot(hist_lv[:,1,0], label="2")
  ax1.plot(hist_lv[:,2,0], label="3")
  ax1.set_ylabel("value")
  ax1.legend()
  plt.savefig("./image/true/lv_%s.png" % name)
  plt.clf()
  plt.close()

def plot_lv_projection(hist_lv, hist_fcst, name, title, nmem):
  # hist_lv   <- np.array[STEPS, DIMM, DIMM]
  # hist_fcst <- np.array[STEPS, nmem, DIMM]
  # name      <- string
  # title     <- string
  # nmem      <- int

  projection = np.zeros((DIMM, nmem))
  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  for i in range(STEPS//2, STEPS):
    for j in range(DIMM):
      for k in range(nmem):
        projection[j,k] += np.abs(np.dot(hist_lv[i,:,j], \
            hist_fcst[i,k,:] - hist_fcst_mean[i,:]))
  projection /= OERR * DIMM * (STEPS / 2.0)

  data = np.log(projection)
  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(data))
  map1 = ax.pcolor(data, cmap=plt.cm.gist_rainbow_r)
  map1.set_clim(-8, -4)
  ax.set_xlim(xmax=nmem)
  # ax.set_aspect(abs(x1-x0)/abs(y1-y0))
  ax.set_xlabel("member")
  ax.set_ylabel("index of vectors")
  cbar = plt.colorbar(map1)
  plt.title(title)
  plt.gca().invert_yaxis()
  plt.savefig("./image/%s/%s.png" % (name, title))
  plt.close()
  return 0

def plot_trajectory_lv(hist_true, hist_lv, name):
  if not (DIMM == 3 or DIMM == 9):
    return 0
  for i_component in range(DIMM//3):
    i_adjust = i_component * 3
    name_component = ["extro", "trop", "ocn"][i_component]

    if (DIMM == 9):
      colors = ["#008000", "#0000ff", "#8080ff", \
                "#80bb80", "#ff0000", "#ff8080", \
                "#000080", "#800000", "#004000"]
    else:
      colors = ["#ff0000", "#008000", "#0000ff"]

    plt.rcParams["font.size"] = 8

    for it in range(0, STEPS, 4):
      itmin = max(0, it - 50)
      fig = plt.figure(figsize=(5.0, 2.5))
      fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, \
        wspace=0.04, hspace=0.04)
      ax1 = fig.add_subplot(121, projection='3d')
      ax2 = fig.add_subplot(122, projection='3d')
      for ax in [ax1, ax2]:
        ax.set_xlim([-30,30])
        ax.set_ylim([-30,30])
        ax.set_zlim([0,50])
        if (ax is ax1):
          ax.view_init(elev=30, azim=-45)
        else:
          ax.view_init(elev=30, azim=-40)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.plot(hist_true[itmin:it+1,0+i_adjust], hist_true[itmin:it+1,1+i_adjust], \
                hist_true[itmin:it+1,2+i_adjust])
        for k in range(DIMM): # LE index
          vector = [hist_true[it,0+i_adjust], hist_true[it,1+i_adjust], \
                    hist_true[it,2+i_adjust], hist_lv[it,k,0+i_adjust], \
                    hist_lv[it,k,1+i_adjust], hist_lv[it,k,2+i_adjust]]
          vec_len = np.sqrt(np.sum(hist_lv[it,k,0+i_adjust:3+i_adjust]**2))
          ax.quiver(*vector, length=(10.0/1.0e-9*vec_len), pivot="tail", color=colors[k])
      plt.savefig("./image/true/tmp_%s_%s_traj_%04d.png" % (name, name_component, it))
      plt.close()

    os.system("convert -delay 5 -loop 0 ./image/true/tmp_*.png \
      ./image/true/%s_lv_%s_traj.gif" % (name, name_component))
    os.system("rm -f image/true/tmp_*.png")
  return 0

def plot_rmse_spread(hist_true, hist_fcst, name, nmem):
  ## refer to a32p23
  # hist_true <- np.array[STEPS, DIMM]
  # hist_fcst <- np.array[STEPS, nmem, DIMM]
  # name <- string
  # nmem <- int

  # Error and Spread_square for each grid and time
  hist_fcst_mean = np.mean(hist_fcst, axis=1)
  hist_fcst_sprd2 = np.zeros((STEPS, DIMM))
  if (nmem > 1):
    for i in range(nmem):
      hist_fcst_sprd2[:,:] = hist_fcst_sprd2[:,:] + \
        1.0 / (nmem - 1.0) * (hist_fcst[:,i,:]**2 - hist_fcst_mean[:,:]**2)
  hist_err = hist_fcst_mean - hist_true

  if (DIMM == 3 or DIMM == 9):
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
      plt.savefig("./image/%s/%s_%s_%s.png" % (name, name, name_component, "time"))
      plt.clf()
      plt.close()

  else: # lorenz96
    # MSE and Spread_square time series (grid average)
    mse_time   = np.mean(hist_err[:,:]**2,     axis=1)
    sprd2_time = np.mean(hist_fcst_sprd2[:,:], axis=1)

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
    plt.title("[%s] RMSE:%6g Spread:%6g" % (name, rmse, sprd))
    plt.savefig("./image/%s/%s_%s.png" % (name, name, "time"))
    plt.clf()
    plt.close()
  return 0

def plot_time_value(hist_true, hist_fcst, hist_obs, name, nmem):
  # name <- string
  # nmem <- int

  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  if (DIMM == 3 or DIMM == 9):
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
      plt.savefig("./image/%s/%s_%s_%s.png" % (name, name, name_component, "val"))
      plt.clf()
      plt.close()
  else:
    plt.rcParams["font.size"] = 12
    fig, ax1 = plt.subplots(1)
    ax1.set_title(name)
    ax1.plot(hist_true[:,0], label="true")
    ax1.plot(hist_fcst_mean[:,0], label="model")
    ax1.plot(hist_obs[:,0], label="obs", linestyle='None', marker=".")
    ax1.set_ylabel("0th element")
    ax1.legend(loc="upper right")
    plt.xlabel("timestep")
    plt.savefig("./image/%s/%s_%s.png" % (name, name, "val"))
    plt.clf()
    plt.close()
  return 0

def plot_3d_trajectory(hist_true, hist_fcst, name, nmem):
  # name <- string
  # nmem <- int

  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  if not (DIMM == 3 or DIMM == 9):
    return 0
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
    plt.savefig("./image/%s/%s_%s_traj.png" % (name, name, name_component))
    plt.clf()
    plt.close()
  return 0

def plot_covariance_matr(hist_covar, name, sel):
  # name <- string

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    rms_covar  = np.sqrt(np.nan_to_num(np.nanmean(hist_covar[STEPS//2:STEPS,:,:]**2, axis=0)))
    mean_covar = np.nan_to_num(np.nanmean(hist_covar, axis=0))
    rms_log = np.log(rms_covar)
  rms_log[np.isneginf(rms_log)] = 0
  plot_matrix(rms_covar , name, "%s_covar_rms_%s"  % (sel, name))
  plot_matrix(rms_log , name, "%s_covar_logrms_%s"  % (sel, name), plt.cm.Reds)
  plot_matrix(mean_covar, name, "%s_covar_mean_%s" % (sel, name))

def plot_matrix(data, name, title, color=plt.cm.bwr):
  # data <- np.array[n,n]
  # name <- string

  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(data))
  map1 = ax.pcolor(data, cmap=color)
  if (color == plt.cm.bwr):
    map1.set_clim(-1.0 * cmax, cmax)
  x0,x1 = ax.get_xlim()
  y0,y1 = ax.get_ylim()
  ax.set_aspect(abs(x1-x0)/abs(y1-y0))
  ax.set_xlabel("x")
  cbar = plt.colorbar(map1)
  plt.title(title)
  plt.gca().invert_yaxis()
  plt.savefig("./image/%s/%s.png" % (name, title))
  plt.close()
  return 0

plot_all()
