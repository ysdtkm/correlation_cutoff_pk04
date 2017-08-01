#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import const
import re

def verif(param1s, param2s, param3s_raw, param3s_arr):
  len1 = len(param1s)
  len2 = len(param2s)
  len3 = len(param3s_raw)
  lenc = 3 # extra, trop, ocean

  rmse_all = np.empty((len1, len2, len3, lenc), dtype=np.float64)
  rmse_all[:,:,:,:] = np.nan
  param3_npa = np.empty((len1, len2, len3, lenc), dtype=np.int32)
  param3_npa[:,:,:,:] = np.nan

  for i, param1 in enumerate(param1s):
    for j, param2 in enumerate(param2s):
      param3s = list(map(int, param3s_arr[i][j]))

      rf = open("image_%s_%s/true/rmse.txt" % (param1, param2), "r")
      for k, line in enumerate(rf):
        xval = int(re.sub(param2 + "_", "", line.split()[0]))
        if param3s[k] != xval:
          raise Exception("param3s mis-specified in super_verif.verif(). param3s[k]: %d, xval: %d" % (param3s[k], xval))
        rmse = list(map(float, line.split()[1:]))
        rmse_all[i,j,k,:] = np.array(rmse[:])
        param3_npa[i,j,k,:] = xval
      rf.close()

  plot_rmse(rmse_all, param1s, param2s, param3_npa)
  plot_min3(rmse_all, param1s, param2s, param3_npa)

  return 0

def plot_rmse(rmse_all, param1s, param2s, param3_npa):
  lenc = 3 # extra, trop, ocean
  colors = ["red", "green", "blue", "yellow", "cyan", "grey"]
  styles = ["solid", "dotted", "dashed", "dashdot"] * 3
  lwidth = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

  for ic in range(lenc):
    name_c = ["extra", "trop", "ocean"][ic]
    for i, param1 in enumerate(param1s):
      plt.rcParams["font.size"] = 8
      fig, ax = plt.subplots()
      fig.subplots_adjust(top=0.85, bottom=0.2, right=0.67)
      for j, param2 in enumerate(param2s):
        x = param3_npa[i,j,:,ic]
        y = rmse_all[i,j,:,ic]
        x_slice = x[~np.isnan(x)]
        y_slice = y[~np.isnan(x)]
        ax.plot(x_slice, y_slice, color=colors[j], label=param2)
      ax.set_ylim(0, max(const.OERR_A, const.OERR_O) * 1.5)
      ax.set_xlabel("Number of yes (assimilated)")
      ax.set_ylabel("RMSE")
      ax.legend(bbox_to_anchor=(1.03,1), loc="upper left")
      plt.title(name_c)
      plt.savefig("verif/verif_%s_%s.pdf" % (param1, name_c))
      plt.clf()
      plt.close()
  return 0

def plot_min3(rmse_all, param1s, param2s, param3_npa):
  # take minimum along param3s, and plot it on param1s-param2s
  lenc = 3 # extra, trop, ocean
  try:
    x = list(map(float, param1s))
  except:
    x = range(len(param1s))
  colors = ["red", "green", "blue", "yellow", "cyan", "grey"]
  styles = ["solid", "dotted", "dashed", "dashdot"] * 3
  lwidth = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
  rmse_min = np.nanmin(rmse_all, axis=2)

  plt.rcParams["font.size"] = 8
  fig, ax = plt.subplots()
  fig.subplots_adjust(top=0.85, bottom=0.2, right=0.67)
  for ic in range(lenc):
    for j, param2 in enumerate(param2s):
      ax.plot(x, rmse_min[:,j,ic], color=colors[ic], linestyle=styles[j], linewidth=lwidth[j],
               label=(["extra", "trop", "ocean"][ic] + "_" + param2))
  ax.set_ylim(0, max(const.OERR_A, const.OERR_O) * 1.5)
  ax.set_xlabel("Ensemble member")
  ax.set_ylabel("minimum RMSE for various number of yes")
  ax.legend(bbox_to_anchor=(1.03,1), loc="upper left")
  plt.savefig("verif/verif_min.pdf")
  plt.clf()
  plt.close()
  return 0

if __name__ == "__main__":
  param1s = ["3", "4", "5", "6"]
  param2s = ["correlation-rms", "covariance-rms"]
  param3s = list(map(str, range(9, 82)))
  verif(param1s, param2s, param3s)

