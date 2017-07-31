#!/usr/bin/env python

import numpy as np
from const import *

def tdvar_b():
  # return -> np.array[DIMM,DIMM] : background error covariance

  if DIMM == 9:
    # obtained by unit_test.py/obtain_tdvar_b(), 8a66120, 100000 timesteps
    b = np.array([
    [ 0.151366652,  0.217993747, 0.00074922173, 0.000654576057, 0.00266310433, -0.00218875723, 0.000794892471, 0.00598052358, -0.00862711731],
    [ 0.217993747,  0.407210895, 0.00539653061, 0.0013996398, 0.00445064454, -0.00249628423, 0.00109589198, 0.00926521126, -0.0124801346],
    [0.00074922173, 0.00539653061,  0.351884549, -0.000379190086, -0.000347220003, 0.000498117183, -0.000366326131, 0.00148373321, -0.00485325959],
    [0.000654576057, 0.0013996398, -0.000379190086, 0.0776620651, 0.0939750707, -0.0179157645, 0.00932743722,  0.109199709, 0.0786195063],
    [0.00266310433, 0.00445064454, -0.000347220003, 0.0939750707,   0.16266338, -0.00415181247, 0.0706870452,  0.255559538,  0.139336958],
    [-0.00218875723, -0.00249628423, 0.000498117183, -0.0179157645, -0.00415181247,  0.188983318, -0.0589431514, -0.173905592, 0.0943891795],
    [0.000794892471, 0.00109589198, -0.000366326131, 0.00932743722, 0.0706870452, -0.0589431514,  0.561413675,   1.10104363,  0.465250176],
    [0.00598052358, 0.00926521126, 0.00148373321,  0.109199709,  0.255559538, -0.173905592,   1.10104363,   4.04932551,  0.227404405],
    [-0.00862711731, -0.0124801346, -0.00485325959, 0.0786195063,  0.139336958, 0.0943891795,  0.465250176,  0.227404405,   3.04948996]
    ])

  elif DIMM == 3:
    b = np.array([ \
      [ 0.34897187,0.5587116,-0.08293288], \
      [ 0.5587116, 1.10975806,0.00229167], \
      [-0.08293288,0.00229167,0.60078791] \
    ])

  else:
    b = np.diag(np.ones(DIMM)) * 1.5

  return b

def stats_order(r_local):
  if DIMM != 9:
    raise Exception("stats_order() is only for 9-variable PK04 model")

  # order_table obtained from 5782bd1, offline.py, 1000TU
  if r_local == "correlation-mean":
    order_table = np.array([
    [ 0, 12, 42, 46, 38, 68, 70, 60, 64],
    [12,  1, 50, 48, 40, 72, 58, 54, 56],
    [42, 50,  2, 80, 78, 52, 76, 62, 74],
    [46, 48, 80,  3, 10, 26, 34, 20, 32],
    [38, 40, 78, 10,  4, 66, 24, 18, 30],
    [68, 72, 52, 26, 66,  5, 28, 36, 22],
    [70, 58, 76, 34, 24, 28,  6, 14, 16],
    [60, 54, 62, 20, 18, 36, 14,  7, 44],
    [64, 56, 74, 32, 30, 22, 16, 44,  8]],
    dtype=np.int32)
  elif r_local == "correlation-rms":
    order_table = np.array([
    [ 0, 10, 18, 64, 48, 56, 68, 66, 70],
    [10,  1, 20, 52, 46, 50, 58, 54, 62],
    [18, 20,  2, 76, 60, 72, 78, 74, 80],
    [64, 52, 76,  3, 14, 22, 42, 36, 28],
    [48, 46, 60, 14,  4, 44, 38, 34, 30],
    [56, 50, 72, 22, 44,  5, 40, 32, 26],
    [68, 58, 78, 42, 38, 40,  6, 16, 12],
    [66, 54, 74, 36, 34, 32, 16,  7, 24],
    [70, 62, 80, 28, 30, 26, 12, 24,  8]],
    dtype=np.int32)
  elif r_local == "covariance-mean":
    order_table = np.array([
    [ 8,  7, 44, 66, 48, 74, 68, 52, 50],
    [ 7,  2, 38, 56, 40, 70, 64, 46, 42],
    [44, 38,  3, 80, 76, 72, 78, 62, 60],
    [66, 56, 80, 24, 19, 34, 36, 26, 21],
    [48, 40, 76, 19, 10, 58, 30, 17, 15],
    [74, 70, 72, 34, 58, 11, 28, 13, 23],
    [68, 64, 78, 36, 30, 28,  9,  5, 32],
    [52, 46, 62, 26, 17, 13,  5,  1, 54],
    [50, 42, 60, 21, 15, 23, 32, 54,  0]],
    dtype=np.int32)
  elif r_local == "covariance-rms":
    order_table = np.array([
    [17, 11, 16, 80, 70, 78, 74, 54, 56],
    [11,  4, 13, 72, 58, 68, 60, 44, 46],
    [16, 13,  5, 76, 64, 66, 62, 52, 50],
    [80, 72, 76, 34, 27, 38, 48, 36, 31],
    [70, 58, 64, 27, 14, 33, 42, 29, 24],
    [78, 68, 66, 38, 33, 25, 40, 22, 20],
    [74, 60, 62, 48, 42, 40, 18,  7,  9],
    [54, 44, 52, 36, 29, 22,  7,  1,  3],
    [56, 46, 50, 31, 24, 20,  9,  3,  0]],
    dtype=np.int32)
  elif r_local == "BHHtRi-mean":
    order_table = np.array([
    [ 4,  3, 36, 56, 40, 63, 79, 71, 70],
    [ 3,  0, 27, 46, 29, 59, 78, 68, 67],
    [36, 27,  1, 75, 66, 61, 80, 77, 76],
    [56, 46, 75, 16, 11, 23, 64, 42, 34],
    [40, 29, 66, 11,  5, 49, 47, 32, 31],
    [63, 59, 61, 23, 49,  6, 44, 30, 37],
    [57, 52, 69, 24, 19, 18, 25, 21, 54],
    [43, 38, 51, 17,  9,  7, 21, 15, 73],
    [41, 33, 50, 12,  8, 13, 54, 73, 14]],
    dtype=np.int32)
  elif r_local == "BHHtRi-rms":
    order_table = np.array([
    [ 9,  3,  8, 67, 54, 64, 80, 76, 77],
    [ 3,  0,  5, 56, 42, 50, 78, 71, 72],
    [ 8,  5,  1, 61, 46, 48, 79, 75, 74],
    [67, 56, 61, 20, 15, 23, 73, 68, 65],
    [54, 42, 46, 15,  6, 19, 70, 62, 58],
    [64, 50, 48, 23, 19, 13, 69, 57, 52],
    [59, 43, 44, 32, 27, 26, 51, 36, 38],
    [39, 30, 34, 21, 16, 11, 36, 25, 29],
    [40, 31, 33, 17, 12, 10, 38, 29, 24]],
    dtype=np.int32)
  elif r_local == "covariance-clim":
    order_table = np.array([
    [25, 24, 70, 60, 56, 64, 42, 38, 48],
    [24, 17, 76, 58, 52, 62, 44, 34, 46],
    [70, 76, 20, 78, 80, 74, 68, 54, 72],
    [60, 58, 78, 29, 28, 66, 40, 14, 12],
    [56, 52, 80, 28, 26, 50, 31,  8, 10],
    [64, 62, 74, 66, 50, 32, 19, 16, 22],
    [42, 44, 68, 40, 31, 19,  4,  3, 36],
    [38, 34, 54, 14,  8, 16,  3,  1,  6],
    [48, 46, 72, 12, 10, 22, 36,  6,  0]],
    dtype=np.int32)
  elif r_local == "correlation-clim":
    order_table = np.array([
    [ 0, 12, 62, 46, 40, 52, 44, 58, 66],
    [12,  1, 70, 42, 38, 54, 50, 56, 68],
    [62, 70,  2, 76, 78, 60, 74, 72, 80],
    [46, 42, 76,  3, 10, 48, 36, 24, 20],
    [40, 38, 78, 10,  4, 34, 28, 16, 22],
    [52, 54, 60, 48, 34,  5, 14, 26, 30],
    [44, 50, 74, 36, 28, 14,  6, 18, 64],
    [58, 56, 72, 24, 16, 26, 18,  7, 32],
    [66, 68, 80, 20, 22, 30, 64, 32,  8]],
    dtype=np.int32)

  return order_table

