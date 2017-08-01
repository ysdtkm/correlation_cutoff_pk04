#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import const

def verif(param1s, param2s, param3s):
  len1 = len(param1s)
  len2 = len(param2s)
  len3 = len(param3s)
  lenc = 3 # extra, trop, ocean

  rmse_all = np.zeros((len1, len2, len3, lenc))
  for i, param1 in enumerate(param1s):
    for j, param2 in enumerate(param2s):
      rf = open("image_%s_%s/true/rmse.txt" % (param1, param2), "r")
      for k, line in enumerate(rf):
        rmse = list(map(float, line.split()[1:]))
        rmse_all[i,j,k:] = np.array(rmse)
      rf.close()

  plot_rmse(rmse_all, param1s, param2s, param3s)
  plot_min3(rmse_all, param1s, param2s, param3s)

  return 0

def plot_rmse(rmse_all, param1s, param2s, param3s):
  lenc = 3 # extra, trop, ocean
  try:
    x = list(map(float, param3s))
  except:
    x = range(len(param3s))
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
        ax.plot(x, rmse_all[i,j,:,ic], color=colors[j], label=param2)
      ax.set_ylim(0, max(const.OERR_A, const.OERR_O) * 1.5)
      ax.set_xlabel("Number of yes (assimilated)")
      ax.set_ylabel("RMSE")
      ax.legend(bbox_to_anchor=(1.03,1), loc="upper left")
      plt.title(name_c)
      plt.savefig("verif/verif_%s_%s.pdf" % (param1, name_c))
      plt.clf()
      plt.close()
  return 0

def plot_min3(rmse_all, param1s, param2s, param3s):
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
  param1s = ["4"]
  param2s = ["covariance-clim"]
  param3s = ["9", "18", "25"]
  verif(param1s, param2s, param3s)

