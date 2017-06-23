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

def test_fdvar_overflow():
  exp = EXPLIST[1]
  h = geth()
  r = getr()
  fcst_0 = np.array( \
    [[-5.443006274201698], \
    [-0.8043972209074476], \
    [29.448353421328868], \
    [-6.875648898446565], \
    [-10.746162351175121], \
    [45.01895925307126], \
    [-40.33122967882807], \
    [-92.74048173257268], \
    [81.4746920842174]] \
  )
  yo = np.array(\
    [[-2.9312942806408193], \
    [-5.972428159803157], \
    [14.373259247649628], \
    [0.14974816307418903], \
    [-3.393558982949476], \
    [47.920238909038034], \
    [-39.34468967151646], \
    [-12.512758026367278], \
    [117.90228464480583]] \
  )
  aint = exp["aint"]

  fdvar(fcst_0, h, r, yo, aint)
  return 0

def test_tangent_model():
  ptb = 1.0e-9
  step_verif = 10

  # initialization (verification t0 -> t1)
  x_t0 = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(STEPS):
    x_t0 = timestep(x_t0, DT)
  x_t1 = np.copy(x_t0)
  for i in range(step_verif):
    x_t1 = timestep(x_t1, DT)

  # get tangent linear
  m = finite_time_tangent(x_t0, DT/1.0, step_verif*1)
  # m = finite_time_tangent_using_nonlinear(x_t0, DT/1.0, step_verif*1)

  sum_sq_diff = 0.0
  for i in range(DIMM):
    # nonlinear perturbation
    x_t0_ptb = np.copy(x_t0)
    x_t0_ptb[i] = x_t0[i] + ptb
    print(i)
    x_t1_ptb = np.copy(x_t0_ptb)
    for j in range(step_verif):
      x_t1_ptb = timestep(x_t1_ptb, DT)

    print("nonlinear:")
    print((x_t1_ptb - x_t1) / ptb)
    print("tangent linear:")
    print(m[:,i].flatten())
    print("diff (NL - TL):")
    diff = (x_t1_ptb - x_t1) / ptb - m[:,i].flatten()
    print(diff)
    sum_sq_diff = sum_sq_diff + np.sum(diff ** 2)

  print("total RMS of diff:")
  print(np.sqrt(sum_sq_diff))
  return 0

def test_tangent_sv():
  step_verif = 10

  x_t0 = np.random.normal(0.0, FERR_INI, DIMM)
  for i in range(STEPS):
    x_t0 = timestep(x_t0, DT)
  m_finite = finite_time_tangent(x_t0, DT/4.0, step_verif*4)
  mt_finite = finite_time_tangent(x_t0, DT/4.0, step_verif*4).T
  eig_vals2, eig_vects2 = np.linalg.eig(m_finite * mt_finite)
  eig_vals3, eig_vects3 = np.linalg.eig(m_finite)
  print("SV growth rates:")
  print(eig_vals2)
  print("LV growth rates:")
  print(eig_vals3)
  print("covariant LVs:")
  print(eig_vects3)
  print("M:")
  print(m_finite)
  print("M.T:")
  print(mt_finite)
  return 0

def test_cost_function_grad():
  if DIMM != 3:
    sys.exit("Set DIMM = 3 for test_cost_function_grad")

  aint = 25
  fcst = np.array([-5.83559367, -6.1021729, 23.42678068])
  anl = np.copy(fcst)
  h = geth([1,1,1])
  r = getr()
  yo = np.array([[-8.27064106], [-1.06064404], [34.80718227]])
  eps = 1.0

  twoj = fdvar_2j(anl, fcst, h, r, yo, aint, 0, DIMM)
  twoj_grad = np.zeros(DIMM)
  for i in range(DIMM):
    anl = np.copy(fcst)
    anl[i] += eps
    twoj_grad[i] = (fdvar_2j(anl, fcst, h, r, yo, aint, 0, DIMM) - twoj) / eps
  print(twoj_grad)

  twoj_grad_anl = fdvar_2j_deriv(anl, fcst, h, r, yo, aint, 0, DIMM)
  print(twoj_grad_anl)

  return 0

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

def plot_matrix(data, name="", title="", color=plt.cm.bwr):
  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(data)) / 100
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
  plt.savefig("./matrix_%s_%s.png" % (name, title))
  plt.close()
  return 0

def check_b():
  new = np.array([ \
  [   31.296179,   44.2633605,  -0.51824455, -0.0765104746,  0.160601432, -0.238184143, 0.0275271624, -0.049991561, 0.0966464721], \
  [  44.2633605,   77.9498178, -0.114422561, -0.0777510266,  0.287639389, -0.428048075, 0.0300395202, -0.103521189,  0.151026639], \
  [ -0.51824455, -0.114422561,   59.3253303, 0.0550755696, 0.0793199082, -0.221180975,  0.101849791,  0.379401384, 0.0187837308], \
  [-0.0765104746, -0.0777510266, 0.0550755696,   5.00294314,   5.63983821,  -2.27517755,  -0.12217024,   1.81940323,  0.954634433], \
  [ 0.160601432,  0.287639389, 0.0793199082,   5.63983821,   10.1444547, 0.0401058189,  0.482065267,    2.7664626,   0.87513345], \
  [-0.238184143, -0.428048075, -0.221180975,  -2.27517755, 0.0401058189,   11.6740375, -0.389886499,  -2.19701294, 0.0501196105], \
  [0.0275271624, 0.0300395202,  0.101849791,  -0.12217024,  0.482065267, -0.389886499,   3.63307352,   7.70722361,   1.91325068], \
  [-0.049991561, -0.103521189,  0.379401384,   1.81940323,    2.7664626,  -2.19701294,   7.70722361,   27.8304653,   0.89669422], \
  [0.0966464721,  0.151026639, 0.0187837308,  0.954634433,   0.87513345, 0.0501196105,   1.91325068,   0.89669422,   18.9551383]  \
  ])

  old = np.array( \
        [[1.55369946e-02, 2.59672759e-02, 2.14143702e-02, 5.50608636e-05, 2.06837903e-04, 9.53104122e-05, 4.01854229e-05, 1.10394688e-04, 1.10795043e-04], \
         [2.59672759e-02, 4.60323124e-02, 3.22253391e-02, 9.67994015e-05, 3.47479991e-04, 1.59885557e-04, 6.25825607e-05, 1.71056662e-04, 1.75287574e-04], \
         [2.14143702e-02, 3.22253391e-02, 4.49870690e-02, 8.07242543e-05, 2.90997346e-04, 1.68933576e-04, 7.17501951e-05, 1.91677973e-04, 2.08898907e-04], \
         [5.50608636e-05, 9.67994015e-05, 8.07242543e-05, 1.19388552e-04, 1.32465565e-04, 1.41518295e-04, 1.26069316e-04, 3.89377305e-04, 5.26723750e-04], \
         [2.06837903e-04, 3.47479991e-04, 2.90997346e-04, 1.32465565e-04, 2.19079467e-04, 1.28264891e-04, 2.15409763e-04, 5.82014129e-04, 7.50531420e-04], \
         [9.53104122e-05, 1.59885557e-04, 1.68933576e-04, 1.41518295e-04, 1.28264891e-04, 3.00276052e-04, 2.90918095e-04, 8.40635972e-04, 8.63515366e-04], \
         [4.01854229e-05, 6.25825607e-05, 7.17501951e-05, 1.26069316e-04, 2.15409763e-04, 2.90918095e-04, 9.88628471e-04, 2.56406108e-03, 2.73643907e-03], \
         [1.10394688e-04, 1.71056662e-04, 1.91677973e-04, 3.89377305e-04, 5.82014129e-04, 8.40635972e-04, 2.56406108e-03, 8.41665768e-03, 6.19483319e-03], \
         [1.10795043e-04, 1.75287574e-04, 2.08898907e-04, 5.26723750e-04, 7.50531420e-04, 8.63515366e-04, 2.73643907e-03, 6.19483319e-03, 1.08456084e-02]] \
      )

  print_two_dim_nparray(old / new * 10000, "%8.4g")
  print_two_dim_nparray(new, "%8.4g")

  plot_matrix(old / new, "", "", color=plt.cm.bwr)
  return 0

obtain_tdvar_b()
