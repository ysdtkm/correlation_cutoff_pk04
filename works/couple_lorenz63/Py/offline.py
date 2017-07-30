#!/usr/bin/env python

import sys
import numpy as np
from const import *
from model import *
from fdvar import *
from main import exec_nature, exec_obs, exec_free_run, exec_assim_cycle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def obtain_climatology():
  nstep = 100000
  all_true = np.empty((nstep, DIMM))

  np.random.seed((10**8+7)*11)
  true = np.random.randn(DIMM) * FERR_INI

  for i in range(0, nstep):
    true[:] = timestep(true[:], DT)
    all_true[i,:] = true[:]
  # all_true.tofile("data/true_for_clim.bin")

  mean = np.mean(all_true[nstep//2:,:], axis=0)
  print("mean")
  print(mean)

  mean2 = np.mean(all_true[nstep//2:,:]**2, axis=0)
  stdv = np.sqrt(mean2 - mean**2)
  print("stdv")
  print(stdv)

  return 0

def obtain_tdvar_b():
  np.random.seed((10**8+7)*12)
  nature = exec_nature()
  obs = exec_obs(nature)
  settings = {"name":"etkf_strong_int8",  "rho":1.1, "aint":8, "nmem":10, \
              "method":"etkf", "couple":"strong"}
  np.random.seed((10**8+7)*13)
  free = exec_free_run(settings)
  anl  = exec_assim_cycle(settings, free, obs)
  hist_bf = np.fromfile("data/%s_covr_back.bin" % settings["name"], np.float64)
  hist_bf = hist_bf.reshape((STEPS, DIMM, DIMM))
  mean_bf = np.nanmean(hist_bf[STEPS//2:, :, :], axis=0)
  trace = np.trace(mean_bf)
  mean_bf *= (DIMM / trace)

  print("[ \\")
  for i in range(DIMM):
    print("[", end="")
    for j in range(DIMM):
      print("%12.9g" % mean_bf[i,j], end="")
      if j < (DIMM - 1):
        print(", ", end="")
    if i < (DIMM - 1):
      print("], \\")
    else:
      print("]  \\")
      print("]")

  return 0

def print_two_dim_nparray(data, format="%12.9g"):
  n = data.shape[0]
  m = data.shape[1]
  print("[ \\")
  for i in range(n):
    print("[", end="")
    for j in range(m):
      print(format % data[i,j], end="")
      if j < (m - 1):
        print(", ", end="")
    if i < (n - 1):
      print("], \\")
    else:
      print("]  \\")
      print("]")

def plot_matrix(data, name="", title="", color=plt.cm.bwr, xlabel="", ylabel="", logscale=False, linthresh=1e-3):
  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(data))
  if logscale:
    map1 = ax.pcolor(data, cmap=color, norm=colors.SymLogNorm(linthresh=linthresh*cmax))
  else:
    map1 = ax.pcolor(data, cmap=color)
  if (color == plt.cm.bwr):
    map1.set_clim(-1.0 * cmax, cmax)
  x0,x1 = ax.get_xlim()
  y0,y1 = ax.get_ylim()
  ax.set_aspect(abs(x1-x0)/abs(y1-y0))
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  cbar = plt.colorbar(map1)
  plt.title(title)
  plt.gca().invert_yaxis()
  plt.savefig("./matrix_%s_%s.png" % (name, title))
  plt.close()
  return 0

def obtain_stats_etkf():
  def cov_to_corr(cov):
    corr = cov.copy()
    for i in range(DIMM):
      for j in range(DIMM):
        corr[i,j] /= np.sqrt(cov[i,i] * cov[j,j])
    return corr

  use_posterior = False

  np.random.seed((10**8+7)*12)
  nature = exec_nature()
  obs = exec_obs(nature)
  settings = {"name":"etkf",  "rho":"adaptive", "nmem":10,
              "method":"etkf", "couple":"strong", "r_local": "full"}
  np.random.seed((10**8+7)*13)
  free = exec_free_run(settings)
  anl  = exec_assim_cycle(settings, free, obs)

  nmem = settings["nmem"]
  hist_fcst = np.fromfile("data/%s_cycle.bin" % settings["name"], np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))

  corr_ijt = np.empty((STEPS, DIMM, DIMM))
  corr_ijt[:,:,:] = np.nan
  cov_ijt = np.empty((STEPS, DIMM, DIMM))
  cov_ijt[:,:,:] = np.nan
  r = getr()

  for it in range(STEPS//2, STEPS):
    if it % AINT == 0:
      if use_posterior:
        fcst = hist_fcst[it, :, :].copy()
      else:
        fcst = hist_fcst[it-AINT, :, :].copy()
        for jt in range(AINT):
          for k in range(nmem):
            fcst[k, :] = timestep(fcst[k,:], DT)
      for i in range(DIMM):
        for j in range(DIMM):
          # a38p40
          vector_i = np.copy(fcst[:, i])
          vector_j = np.copy(fcst[:, j])
          vector_i[:] -= np.mean(vector_i)
          vector_j[:] -= np.mean(vector_j)
          numera = np.sum(vector_i * vector_j)
          denomi = (np.sum(vector_i ** 2) * np.sum(vector_j ** 2)) ** 0.5
          corr_ijt[it, i, j] = numera / denomi
          cov_ijt[it, i, j] = numera / (nmem - 1.0)
      cov_instant_ij = cov_ijt[it, :, :].copy()

  corr_mean_ij   = np.nanmean(corr_ijt, axis=0)
  corr_rms_ij    = np.sqrt(np.nanmean(corr_ijt**2, axis=0))
  cov_mean_ij    = np.nanmean(cov_ijt, axis=0)
  cov_rms_ij     = np.sqrt(np.nanmean(cov_ijt**2, axis=0))
  ri             = np.linalg.inv(getr())
  bhhtri_rms_ij  = cov_rms_ij.dot(ri)
  bhhtri_mean_ij = cov_mean_ij.dot(ri)
  rand_ij        = np.random.randn(DIMM, DIMM)

  clim_mean = np.mean(nature[STEPS//2:,:], axis=0)
  cov_clim_ij = np.empty((DIMM, DIMM))
  k = 0
  for it in range(STEPS//2, STEPS):
    anom = nature[it,:] - clim_mean[:]
    for i in range(DIMM):
      cov_clim_ij[i,:] += anom[i] * anom[:]
    k += 1
  cov_clim_ij[:,:] /= (k - 1.0)
  corr_clim_ij = cov_to_corr(cov_clim_ij)

  data_hash = {"correlation-mean":corr_mean_ij, "correlation-rms":corr_rms_ij, "covariance-mean":cov_mean_ij,
               "covariance-rms":cov_rms_ij, "BHHtRi-mean":bhhtri_mean_ij, "BHHtRi-rms":bhhtri_rms_ij,
               "covariance-clim":cov_clim_ij, "correlation-clim":corr_clim_ij, "random":rand_ij,
               "covariance-instant":cov_instant_ij}
  for name in data_hash:
    plot_matrix(data_hash[name], title=name, xlabel="grid index i", ylabel="grid index j", logscale=True, linthresh=1e-1)
    print(name)
    matrix_order(np.abs(data_hash[name]))

  return 0

def matrix_order(mat_ij, prioritize_diag=False, max_odr=81):
  n = len(mat_ij)
  if len(mat_ij[0]) != n:
    raise Exception("input matrix non-square")

  def find_order(k, l, order, asymmetric):
    if asymmetric:
      i = min(k, l)
      j = max(k, l)
    else:
      i = k
      j = l

    for io, cmp in enumerate(order):
      if cmp[0] == i and cmp[1] == j:
        if (io < len(order) - 1) and (order[io+1][0] == j) and (order[io+1][1] == i):
          # if same elements in symmetric position, return rouded-up number
          return io + 1
        else:
          return io
    raise Exception("find_order overflow")

  def print_order(order):
    for i in range(n):
      print("[", end="")
      for j in range(n):
        if order[i][j] < max_odr:
          print("%2d" % order[i][j], end="")
        else:
          print("**", end="")
        if j < n-1:
          print(", ", end="")
      print("],")
    print("")
    return

  order = [[0 for j in range(n)] for i in range(n)]
  if prioritize_diag:
    upper_components = []
    for i in range(n):
      for j in range(i+1, n):
        upper_components.append((i, j, mat_ij[i,j]))
    order_obj_upper = sorted(upper_components, key=lambda x: x[2], reverse=True)
    for i in range(n):
      for j in range(n):
        if i == j:
          order[i][j] = i
        else:
          order[i][j] = find_order(i, j, order_obj_upper, True) * 2 + 9 + 1
  else:
    all_components = []
    for i in range(n):
      for j in range(n):
        all_components.append((i, j, mat_ij[i,j]))
    order_obj = sorted(all_components, key=lambda x: x[2], reverse=True)
    for i in range(n):
      for j in range(n):
        order[i][j] = find_order(i, j, order_obj, False)

  print_order(order)

  return 0

obtain_stats_etkf()
