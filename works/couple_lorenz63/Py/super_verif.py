#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

  plot(rmse_all, param1s, param2s, param3s)
  try:
    plot_min3(rmse_all, param1s, param2s, param3s)
  except:
    print("if param1s is not number, plot_min3 is skipped")

  return 0

def plot(rmse_all, param1s, param2s, param3s):
  lenc = 3 # extra, trop, ocean
  x = list(map(float, param3s))
  colors = ["r", "g", "b"]
  styles = ["solid", "dotted", "dashed", "dashdot"]
  for i, param1 in enumerate(param1s):
    for ic in range(lenc):
      for j, param2 in enumerate(param2s):
        plt.plot(x, rmse_all[i,j,:,ic], color=colors[ic], linestyle=styles[j], label=(["extra", "trop", "ocean"][ic] + "_" + param2))
    plt.legend()
    plt.savefig("verif/verif_%s.pdf" % param1)
    plt.clf()
    plt.close()
  return 0

def plot_min3(rmse_all, param1s, param2s, param3s):
  # take minimum along param3s, and plot it on param1s-param2s
  lenc = 3 # extra, trop, ocean
  x = list(map(float, param1s))
  colors = ["r", "g", "b"]
  styles = ["solid", "dotted", "dashed", "dashdot"]
  rmse_min = np.mean(rmse_all, axis=2)
  for ic in range(lenc):
    for j, param2 in enumerate(param2s):
      plt.plot(x, rmse_min[:,j,ic], color=colors[ic], linestyle=styles[j],
               label=(["extra", "trop", "ocean"][ic] + "_" + param2))
  plt.legend()
  plt.savefig("verif/verif_min.pdf")
  plt.clf()
  plt.close()
  return 0

if __name__ == "__main__":
  param1s = ["full", "3-components"]
  param2s = ["1.05", "1.1"]
  verif(param1s, param2s)

