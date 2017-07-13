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

  fdvar(fcst_0, h, r, yo, AINT)
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

def plot_matrix(data, name="", title="", color=plt.cm.bwr, xlabel="", ylabel=""):
  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
  cmax = np.max(np.abs(data))
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

def check_b():
  # obtained by unit_test.py/obtain_tdvar_b(), a66aa31, 100000 timesteps
  new = np.array([ \
    [  29.1361601,   40.8007041, 0.0996813749, 0.0791912453,  0.344864776, -0.136810502, 0.00336997879, 0.0318867672, 0.0437722107], \
    [  40.8007041,   72.4114206,   0.90879686,  0.159329634,  0.564511249, -0.195910748, 0.00951459272, 0.0774520914, 0.0487791362], \
    [0.0996813749,   0.90879686,   56.1769923, 0.0200836456, 0.0333094552, 0.00326933485, 0.00874575146, -0.00783791119, 0.0750019486], \
    [0.0791912453,  0.159329634, 0.0200836456,   4.82189397,   5.65114453,  -1.76995579, -0.189879111,    1.8832435,  0.619351491], \
    [ 0.344864776,  0.564511249, 0.0333094552,   5.65114453,   10.1145373,  0.202050997,  0.339100741,   2.79224997,  0.648284682], \
    [-0.136810502, -0.195910748, 0.00326933485,  -1.76995579,  0.202050997,    11.011319,  -0.20636934,  -1.39320264, 0.0335684046], \
    [0.00336997879, 0.00951459272, 0.00874575146, -0.189879111,  0.339100741,  -0.20636934,   3.56934671,   7.00919827,   2.82018057], \
    [0.0318867672, 0.0774520914, -0.00783791119,    1.8832435,   2.79224997,  -1.39320264,   7.00919827,   26.5619341,   1.63512268], \
    [0.0437722107, 0.0487791362, 0.0750019486,  0.619351491,  0.648284682, 0.0335684046,   2.82018057,   1.63512268,   19.8071293]  \
  ])

  # not known how it was obtained. trace ~= 127
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
  ) * 1000

  print_two_dim_nparray(old / new * 10000, "%8.4g")
  print_two_dim_nparray(new, "%8.4g")
  print(np.trace(old))

  plot_matrix(np.log(np.abs(old / new)), "", "old_over_new", color=plt.cm.Reds)
  plot_matrix(np.log(np.abs(old)), "", "old", color=plt.cm.Reds)
  plot_matrix(np.log(np.abs(new)), "", "new", color=plt.cm.Reds)
  return 0

def compare_coupled_vs_persistent_bc():
  nstep = 10000
  leadtime = 100

  np.random.seed(100000007*2)
  all_true = np.empty((nstep, DIMM))
  true = np.random.normal(0.0, FERR_INI, DIMM)
  fcst_cp = np.empty((DIMM))
  fcst_bc = np.empty((DIMM))
  msd_extra = np.zeros((leadtime))
  msd_trop = np.zeros((leadtime))
  msd_ocean = np.zeros((leadtime))

  # forward integration i-1 -> i
  for i in range(nstep):
    true[:] = timestep(true[:], DT)
    all_true[i,:] = true[:]

  fcst_interval = 100
  for i in range(nstep // fcst_interval):
    persis_bc = all_true[i*fcst_interval,:].copy()
    fcst_cp[:] = all_true[i*fcst_interval,:]
    fcst_bc[:] = all_true[i*fcst_interval,:]

    for j in range(leadtime):
      fcst_cp[:] = timestep(fcst_cp[:], DT)
      fcst_bc[0:6] = timestep(fcst_bc[0:6], DT, 0, 6, persis_bc)
      fcst_bc[6:9] = timestep(fcst_bc[6:9], DT, 6, 9, persis_bc)

      msd_extra[j] += np.mean((fcst_cp[0:3] - fcst_bc[0:3])**2)
      msd_trop[j]  += np.mean((fcst_cp[3:6] - fcst_bc[3:6])**2)
      msd_ocean[j] += np.mean((fcst_cp[6:9] - fcst_bc[6:9])**2)

  msd_extra /= (nstep / fcst_interval)
  msd_trop  /= (nstep / fcst_interval)
  msd_ocean /= (nstep / fcst_interval)

  for j in range(leadtime):
    print(j+1, msd_extra[j]**0.5, msd_trop[j]**0.5, msd_ocean[j]**0.5)

  plt.plot(range(1, leadtime+1), msd_extra[:]**0.5, label="extra")
  plt.plot(range(1, leadtime+1), msd_trop[:]**0.5, label="trop")
  plt.plot(range(1, leadtime+1), msd_ocean[:]**0.5, label="ocean")
  plt.xlim(0, leadtime)
  plt.ylim(0, 8)
  plt.axhline(y = OERR_A)
  plt.axvline(x = 8)
  plt.axvline(x = 25)
  plt.legend()
  plt.xlabel("forecast leadtime (steps)")
  plt.ylabel("RMSD, coupled vs persistent BC forecasts")
  plt.savefig("./rmsd_coupled_vs_persistentbc.png")

def obtain_r2_etkf():
  np.random.seed(100000007*2)
  nature = exec_nature()
  obs = exec_obs(nature)
  settings = {"name":"etkf_strong_int8",  "rho":1.1, "nmem":10,
              "method":"etkf", "couple":"strong", "r_local": "full"}
  np.random.seed(100000007*3)
  free = exec_free_run(settings)
  anl  = exec_assim_cycle(settings, free, obs)

  nmem = settings["nmem"]
  hist_fcst = np.fromfile("data/%s_cycle.bin" % settings["name"], np.float64)
  hist_fcst = hist_fcst.reshape((STEPS, nmem, DIMM))

  r2_ijt = np.empty((STEPS, DIMM, DIMM))
  r2_ijt[:,:,:] = np.nan
  for it in range(STEPS//2, STEPS):
    if it % AINT == 0:
      # reproduce background
      fcst = np.copy(hist_fcst[it-AINT, :, :])
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
          # numera = np.sum(vector_i * vector_j) ** 2
          # denomi = np.sum(vector_i ** 2) * np.sum(vector_j ** 2)
          numera = np.sum(vector_i * vector_j)
          denomi = (np.sum(vector_i ** 2) * np.sum(vector_j ** 2)) ** 0.5
          r2 = numera / denomi
          r2_ijt[it, i, j] = np.copy(r2)
  r2_ij = np.nanmean(r2_ijt, axis=0)
  print(r2_ij)
  plot_matrix(r2_ij, title="R_squared", xlabel="grid index i", ylabel="grid index j")
  return 0

obtain_r2_etkf()
