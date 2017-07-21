#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
from mpl_toolkits.mplot3d import Axes3D
from const import *

def plot_all():
  Plot_3d = True

  hist_true = np.fromfile("data/true.bin", np.float64)
  hist_true = hist_true.reshape((STEPS, DIMM))
  if Calc_lv:
    vectors = ["blv", "flv", "clv", "fsv", "isv"]
  else:
    vectors = []
  vector_name = {"blv": "Backward_LV", "flv": "Forward_LV", "clv": "Characteristic_LV", \
                 "fsv": "Final_SV", "isv": "Initial_SV"}
  hist_vector = {}
  for vec in vectors:
    hist_vector[vec] = np.fromfile("data/%s.bin" % vec, np.float64)
    hist_vector[vec] = hist_vector[vec].reshape((STEPS, DIMM, DIMM))

  os.system("mkdir -p image/true")

  if Calc_lv:
    plot_le()
  for vec in vectors:
    plot_lv_time(hist_vector[vec], vector_name[vec])
    # plot_trajectory_lv(hist_true, hist_vector[vec], vector_name[vec])

  global rmse_hash
  rmse_hash = {}

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
    if Plot_3d:
      plot_3d_trajectory(hist_true, hist_fcst, name, nmem)

    if (exp["method"] == "etkf"):
      if exp["rho"] == "adaptive" or exp["rho"] == "adaptive_each":
        plot_adaptive_inflation(name, exp["rho"])

      for vec in vectors:
        is_oblique = (vec == "clv")
        plot_lv_projection(hist_vector[vec], hist_fcst, name, vector_name[vec], nmem, is_oblique)

      for sel in ["back", "anl"]:
        hist_covar = np.fromfile("data/%s_covr_%s.bin" % (name, sel), np.float64)
        hist_covar = hist_covar.reshape((STEPS, DIMM, DIMM))
        plot_covariance_matr(hist_covar, name, sel)
  plot_rmse_bar(hist_true)

def plot_le():
  # plot lyapunov exponents
  hist_ble = np.fromfile("data/ble.bin", np.float64)
  hist_fle = np.fromfile("data/fle.bin", np.float64)
  hist_fse = np.fromfile("data/fse.bin", np.float64)
  hist_ise = np.fromfile("data/ise.bin", np.float64)

  hist_ble = hist_ble.reshape((STEPS, DIMM))
  hist_fle = hist_fle.reshape((STEPS, DIMM))
  hist_fse = hist_fse.reshape((STEPS, DIMM))
  hist_ise = hist_ise.reshape((STEPS, DIMM))

  mean_ble = np.mean(hist_ble[STEPS//4:(STEPS*3)//4,:], axis=0)
  mean_fle = np.mean(hist_fle[STEPS//4:(STEPS*3)//4,:], axis=0)
  mean_fse = np.mean(hist_fse[STEPS//4:(STEPS*3)//4,:], axis=0)
  mean_ise = np.mean(hist_ise[STEPS//4:(STEPS*3)//4,:], axis=0)

  plt.rcParams["font.size"] = 12
  fig, ax1 = plt.subplots(1)
  ax1.set_title("Lyapunov exponents")
  ax1.plot(mean_ble, label="Backward Lyap.")
  ax1.plot(mean_fle, label="Forward Lyap.")
  ax1.plot(mean_fse, label="mean Singular")
  # ax1.plot(mean_ise, label="Initial Singular")
  ax1.set_ylabel("1 / Time")
  ax1.set_xlabel("LE index")
  ax1.legend()
  plt.savefig("./image/true/le.pdf")
  plt.clf()
  plt.close()

def plot_lv_time(hist_lv, name):
  # hist_lv <- np.array[STEPS, DIMM, DIMM]
  # name    <- string
  plt.rcParams["font.size"] = 12
  fig, ax1 = plt.subplots(1)
  ax1.set_title("lv1_1")
  ax1.plot(hist_lv[:,0,0], label="1")
  ax1.plot(hist_lv[:,1,0], label="2")
  ax1.plot(hist_lv[:,2,0], label="3")
  ax1.set_ylabel("value")
  ax1.legend()
  plt.savefig("./image/true/lv_%s.pdf" % name)
  plt.clf()
  plt.close()

def plot_lv_projection(hist_lv, hist_fcst, name, title, nmem, is_oblique):
  # hist_lv    <- np.array[STEPS, DIMM, DIMM]
  # hist_fcst  <- np.array[STEPS, nmem, DIMM]
  # name       <- string
  # title      <- string
  # nmem       <- int
  # is_oblique <- bool

  projection = np.zeros((DIMM, nmem))
  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  for i in range(STEPS//4, (STEPS*3)//4):
    if is_oblique:
      for k in range(nmem):
        projection[:,k] += np.abs(oblique_projection(hist_fcst[i,k,:] - hist_fcst_mean[i,:], hist_lv[i,:,:]))
    else:
      for j in range(DIMM):
        for k in range(nmem):
          projection[j,k] += np.abs(np.dot(hist_lv[i,:,j], \
              hist_fcst[i,k,:] - hist_fcst_mean[i,:]))
  projection /= (STEPS / 2.0)

  data = np.log(projection)
  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(data))
  map1 = ax.pcolor(data, cmap=plt.cm.jet)
  if is_oblique:
    map1.set_clim(-3.5, 1.5)
  else:
    map1.set_clim(-5, 0)
  ax.set_xlim(xmax=nmem)
  ax.set_xlabel("member")
  ax.set_ylabel("index of vectors")
  cbar = plt.colorbar(map1)
  plt.title(title)
  plt.gca().invert_yaxis()
  plt.savefig("./image/%s/%s.pdf" % (name, title))
  plt.close()
  return 0

def plot_trajectory_lv(hist_true, hist_lv, name):
  # hist_true <- np.array[STEPS, DIMM]
  # hist_lv   <- np.array[STEPS, DIMM, DIMM]
  # name      <- string

  if not (DIMM == 3 or DIMM == 9):
    return 0
  for i_component in range(DIMM//3):
    i_adjust = i_component * 3
    name_component = ["extra", "trop", "ocean"][i_component]

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
      plt.savefig("./image/true/tmp_%s_%s_traj_%04d.pdf" % (name, name_component, it))
      plt.close()

    os.system("convert -delay 5 -loop 0 ./image/true/tmp_*.pdf \
      ./image/true/%s_lv_%s_traj.gif" % (name, name_component))
    os.system("rm -f image/true/tmp_*.pdf")
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

  global rmse_hash

  if (DIMM == 3 or DIMM == 9):
    rmse_component = []
    for i_component in range(DIMM//3):
      grid_from = 3 * i_component
      name_component = ["extra", "trop", "ocean"][i_component]

      # MSE and Spread_square time series (grid average)
      mse_time   = np.mean(hist_err[:,grid_from:grid_from+3]**2,     axis=1)
      sprd2_time = np.mean(hist_fcst_sprd2[:,grid_from:grid_from+3], axis=1)

      # RMSE and Spread time average
      rmse = np.sqrt(np.mean(mse_time[STEPS//2:]))
      sprd = np.sqrt(np.mean(sprd2_time[STEPS//2:]))
      rmse_component.append(rmse)

      # RMSE-Spread time series
      plt.rcParams["font.size"] = 14
      plt.yscale('log')
      if i_component == 2:
        plt.axhline(y=OERR_O, label="sqrt(R)", alpha=0.5)
      else:
        plt.axhline(y=OERR_A, label="sqrt(R)", alpha=0.5)
      plt.plot(np.sqrt(mse_time), label="RMSE")
      if (nmem > 1):
        plt.plot(np.sqrt(sprd2_time), label="Spread")
      plt.legend()
      plt.xlabel("timestep")
      plt.ylim([0.01, 100])
      plt.title("[%s %s] RMSE:%6.4g Spread:%6.4g" % (name, name_component, rmse, sprd))
      plt.savefig("./image/%s/%s_%s_%s.pdf" % (name, name, name_component, "time"), dpi=80)
      plt.clf()
      plt.close()

    # text output
    f = open("./image/true/rmse.txt", "a")
    f.write(("%-25s" % name) + " ".join(map(str, rmse_component)) + "\n")
    f.close()

    f = open("./image/true/rmse_for_tex.txt", "a")
    f.write(("%-25s" % name) + " ".join(["%-8.03g" % x for x in rmse_component]) + "\n")
    f.close()

    print(("%-25s" % name) + " ".join(["%-8.03g" % x for x in rmse_component]))
    rmse_hash[name] = rmse_component

  else: # lorenz96
    # MSE and Spread_square time series (grid average)
    mse_time   = np.mean(hist_err[:,:]**2,     axis=1)
    sprd2_time = np.mean(hist_fcst_sprd2[:,:], axis=1)

    # RMSE and Spread time average
    rmse = np.sqrt(np.mean(mse_time[STEPS//2:]))
    sprd = np.sqrt(np.mean(sprd2_time[STEPS//2:]))

    # RMSE-Spread time series
    plt.rcParams["font.size"] = 16
    plt.yscale('log')
    plt.plot(np.sqrt(mse_time), label="RMSE")
    if (nmem > 1):
      plt.plot(np.sqrt(sprd2_time), label="Spread")
    plt.legend()
    plt.xlabel("timestep")
    plt.title("[%s] RMSE:%6g Spread:%6g" % (name, rmse, sprd))
    plt.savefig("./image/%s/%s_%s.pdf" % (name, name, "time"))
    plt.clf()
    plt.close()
  return 0

def plot_time_value(hist_true, hist_fcst, hist_obs, name, nmem):
  # hist_true  <- np.array[STEPS, DIMM]
  # hist_fcst  <- np.array[STEPS, nmem, DIMM]
  # hist_obs   <- np.array[STEPS, DIMO]
  # name       <- string
  # nmem       <- int

  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  if (DIMM == 3 or DIMM == 9):
    for i_component in range(DIMM//3):
      i_adjust = i_component * 3
      name_component = ["extra", "trop", "ocean"][i_component]

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
      plt.savefig("./image/%s/%s_%s_%s.pdf" % (name, name, name_component, "val"))
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
    plt.savefig("./image/%s/%s_%s.pdf" % (name, name, "val"))
    plt.clf()
    plt.close()
  return 0

def plot_3d_trajectory(hist_true, hist_fcst, name, nmem):
  # hist_true  <- np.array[STEPS, DIMM]
  # hist_fcst  <- np.array[STEPS, nmem, DIMM]
  # name <- string
  # nmem <- int

  hist_fcst_mean = np.mean(hist_fcst, axis=1)

  if not (DIMM == 3 or DIMM == 9):
    return 0
  for i_component in range(DIMM//3):
    i_adjust = i_component * 3
    name_component = ["extra", "trop", "ocean"][i_component]

    # 3D trajectory
    plt.rcParams["font.size"] = 16
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, \
      wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(111, projection='3d')
    st = STEPS // 2 # step to start plotting
    ax.plot(hist_true[st:,0+i_adjust], hist_true[st:,1+i_adjust], \
      hist_true[st:,2+i_adjust], label="true")
    ax.plot(hist_fcst_mean[st:,0+i_adjust], hist_fcst_mean[st:,1+i_adjust], \
      hist_fcst_mean[st:,2+i_adjust], label="model")
    ax.legend()
    # ax.set_xlim([-30,30])
    # ax.set_ylim([-30,30])
    # ax.set_zlim([0,50])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("./image/%s/%s_%s_traj.pdf" % (name, name, name_component))
    plt.clf()
    plt.close()
  return 0

def plot_covariance_matr(hist_covar, name, sel):
  # hist_covar <- np.array[STEPS, DIMM, DIMM]
  # name       <- string
  # sel        <- string

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    rms_covar  = np.sqrt(np.nan_to_num(np.nanmean(hist_covar[STEPS//2:STEPS,:,:]**2, axis=0)))
    mean_covar = np.nan_to_num(np.nanmean(hist_covar[STEPS//2:STEPS,:,:], axis=0))
    rms_log = np.log(rms_covar)
    mean_cosine = np.copy(mean_covar)
    for i in range(DIMM):
      mean_cosine[i,:] /= np.sqrt(mean_covar[i,i])
      mean_cosine[:,i] /= np.sqrt(mean_covar[i,i])

  rms_log[np.isneginf(rms_log)] = 0
  plot_matrix(rms_covar , name, "%s_%s_covar_rms"  % (name, sel))
  plot_matrix(rms_log , name, "%s_%s_covar_logrms"  % (name, sel), plt.cm.Reds)
  plot_matrix(mean_covar, name, "%s_%s_covar_mean" % (name, sel))
  plot_matrix(mean_cosine, name, "%s_%s_cosine_mean" % (name, sel))

def plot_matrix(data, name, title, color=plt.cm.bwr):
  # data  <- np.array[n,n]
  # name  <- string
  # title <- string
  # color <- plt.cm object

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
  plt.savefig("./image/%s/%s.pdf" % (name, title))
  plt.close()
  return 0

def oblique_projection(vector, obl_basis):
  # vector    <- np.array[DIMM]
  # obl_basis <- np.array[DIMM, DIMM]
  # return    -> np.array[DIMM]

  if not ((obl_basis.shape[0] == obl_basis.shape[1]) and (obl_basis.shape[0] == vector.shape[0])):
    print(vector.shape)
    print(obl_basis.shape)
    sys.exit("oblique projection: only square matrix is arrowed as obl_basis")

  q, r = np.linalg.qr(obl_basis)

  orth_coefs = np.dot(q.T, vector)
  try:
    coefs = np.dot(np.linalg.inv(r), orth_coefs)
  except np.linalg.linalg.LinAlgError:
    coefs = np.zeros(DIMM)

  return coefs

def plot_rmse_bar(hist_true):
  if DIMM != 9:
    return 0

  global rmse_hash
  plt.rcParams["font.size"] = 9

  nexp = len(rmse_hash)
  width = 1.0 / (nexp + 1)

  fig, ax = plt.subplots()
  fig.subplots_adjust(top=0.85, bottom=0.2, right=0.67)
  oerr_a = ax.axhline(y=OERR_A, label="sqrt(R_atmos)", alpha=0.5, color="red")
  oerr_o = ax.axhline(y=OERR_O, label="sqrt(R_ocean)", alpha=0.5, color="blue")

  plist = []
  j = 0
  for name in rmse_hash:
    shift = width * j
    x = [(i + shift) for i in range(3)]
    p = ax.bar(x, rmse_hash[name], width, label=name )
    plist.append(p)
    j += 1

  ax.set_ylim(0, max(OERR_O, OERR_A)*1.5)
  ax.set_xticks([(i + width * (nexp - 1) * 0.5) for i in range(3)])
  ax.set_xticklabels(["extra", "trop", "ocean"], rotation = 0)
  ax.set_ylabel("RMSE")

  plist += [oerr_a, oerr_o]
  ax.legend(plist, [i.get_label() for i in plist], bbox_to_anchor=(1.03,1), loc="upper left")
  plt.savefig("./image/true/rmse_bar.pdf")

  return 0

def plot_adaptive_inflation(name, method):
  hist_infl = np.fromfile("data/%s_inflation.bin" % name, np.float64)
  hist_infl = hist_infl.reshape((STEPS, 3))

  if method == "adaptive_each":
    for i in range(3):
      name_component = ["extra", "trop", "ocean"][i]
      color = ["r", "g", "b"][i]
      plt.plot(hist_infl[:,i], color=color, label=name_component)
  else:
    plt.plot(hist_infl[:,0], label="common")

  plt.xlim(0, STEPS)
  plt.ylim(0.9, 1.2)
  plt.axhline(y=1.0, color="black", alpha=0.5)
  plt.title("adaptive inflation")
  plt.legend()
  plt.savefig("./image/%s/%s_inflation.pdf" % (name, name))
  plt.close()

plot_all()
