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

  np.random.seed(1)
  true = np.random.normal(0.0, FERR_INI, DIMM)

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
  np.random.seed(100000007*2)
  nature = exec_nature()
  obs = exec_obs(nature)
  settings = {"name":"etkf_strong_int8",  "rho":1.1, "aint":8, "nmem":10, \
              "method":"etkf", "couple":"strong"}
  np.random.seed(100000007*3)
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

def obtain_r2_etkf():
  use_posterior = False

  np.random.seed(100000007*2)
  nature = exec_nature()
  obs = exec_obs(nature)
  settings = {"name":"etkf_strong_int8",  "rho":"adaptive", "nmem":10,
              "method":"etkf", "couple":"strong", "r_local": "full"}
  np.random.seed(100000007*3)
  free = exec_free_run(settings)
  anl  = exec_assim_cycle(settings, free, obs)

  nmem = settings["nmem"]
  hist_fcst = np.fromfile("data/%s_cycle.bin" % settings["name"], np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))

  corr_ijt = np.empty((STEPS, DIMM, DIMM))
  corr_ijt[:,:,:] = np.nan
  corr2_ijt = np.empty((STEPS, DIMM, DIMM))
  corr2_ijt[:,:,:] = np.nan
  cov_ijt = np.empty((STEPS, DIMM, DIMM))
  cov_ijt[:,:,:] = np.nan
  cov2_ijt = np.empty((STEPS, DIMM, DIMM))
  cov2_ijt[:,:,:] = np.nan

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
          corr2_ijt[it, i, j] = numera ** 2 / denomi ** 2
          cov_ijt[it, i, j] = numera
          cov2_ijt[it, i, j] = numera ** 2


  corr_mean_ij = np.nanmean(corr_ijt, axis=0)
  corr_rms_ij = np.sqrt(np.nanmean(corr2_ijt, axis=0))
  cov_mean_ij = np.nanmean(cov_ijt, axis=0)
  cov_rms_ij = np.sqrt(np.nanmean(cov2_ijt, axis=0))
  ri = np.linalg.inv(getr())
  bhhtri_ij = np.abs(cov_rms_ij.dot(ri))

  plot_matrix(corr_mean_ij, title="Corr_mean", xlabel="grid index i", ylabel="grid index j", logscale=True, linthresh=1e-2)
  plot_matrix(corr_rms_ij, title="Corr_rms", xlabel="grid index i", ylabel="grid index j", logscale=True, linthresh=1e-2)
  plot_matrix(cov_mean_ij, title="Cov_mean", xlabel="grid index i", ylabel="grid index j", logscale=True, linthresh=1e-3)
  plot_matrix(cov_rms_ij, title="Cov_rms", xlabel="grid index i", ylabel="grid index j", logscale=True, linthresh=1e-3)
  plot_matrix(bhhtri_ij, title="BHHtRi", xlabel="grid index i", ylabel="grid index j", logscale=True, linthresh=1e-3)

  print("correlation-mean")
  matrix_nondiagonal_order(corr_mean_ij)
  print("correlation-rms")
  matrix_nondiagonal_order(corr_rms_ij)
  print("covariance-mean")
  matrix_nondiagonal_order(cov_mean_ij)
  print("covariance-rms")
  matrix_nondiagonal_order(cov_rms_ij)
  print("BHHtRi")
  matrix_nondiagonal_order(bhhtri_ij)
  print("random")
  mat_rand = np.random.randn(DIMM, DIMM)
  matrix_nondiagonal_order(mat_rand)
  return 0

def matrix_nondiagonal_order(mat_ij, prioritize_diag=False, max_odr=81):
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
        return io
    raise Exception("find_order overflow")

  if prioritize_diag:
    nondiagonal_components = []
    for i in range(n):
      for j in range(i+1, n):
        nondiagonal_components.append((i, j, mat_ij[i,j]))
    order = sorted(nondiagonal_components, key=lambda x: x[2], reverse=True)

    for i in range(n):
      for j in range(n):
        if i == j:
          print("%2d " % i, end="")
        else:
          odr = find_order(i, j, order, True) * 2 + 9
          if odr < max_odr:
            print("%2d " % odr, end="")
          else:
            print("** ", end="")
      print("")
    print("")

  else:
    all_components = []
    for i in range(n):
      for j in range(n):
        all_components.append((i, j, mat_ij[i,j]))
    order = sorted(all_components, key=lambda x: x[2], reverse=True)

    for i in range(n):
      for j in range(n):
        odr = find_order(i, j, order, False)
        if odr < max_odr:
          print("%2d " % odr, end="")
        else:
          print("** ", end="")
      print("")
    print("")

  return 0

obtain_r2_etkf()
