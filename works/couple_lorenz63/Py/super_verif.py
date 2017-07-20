#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def verif(param1s, param2s):
  len1 = len(param1s)
  len2 = len(param2s)
  lenc = 3 # extra, trop, ocean

  rmse_all = np.zeros((len1, len2, lenc))
  for i, param1 in enumerate(param1s):
    for j, param2 in enumerate(param2s):
      rf = open("image_%s_%s/true/rmse.txt" % (param1, param2), "r")
      for line in rf:
        rmse = list(map(float, line.split()[1:]))
        rmse_all[i,j,:] = np.array(rmse)
      rf.close()

  x = list(map(float, param2s))
  colors = ["r", "g", "b"]
  styles = ["solid", "dotted", "dashed", "dashdot"]
  for ic in range(lenc):
    for i, param1 in enumerate(param1s):
      plt.plot(x, rmse_all[i,:,ic], color=colors[ic], linestyle=styles[i], label=(["extra", "trop", "ocean"][ic] + "_" + param1))
  plt.legend()
  plt.savefig("verif/verif.pdf")
  return 0

if __name__ == "__main__":
  param1s = ["full", "3-components"]
  param2s = ["1.05", "1.1"]
  verif(param1s, param2s)

