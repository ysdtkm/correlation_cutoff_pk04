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

  # order_table obtained from d265ebb, unit_test.py (diagonal elements prioritized)
  if r_local == "correlation-mean":
    order_table = np.array([
    [ 0,  9, 49, 57, 39, 41, 71, 73, 51],
    [10,  1, 53, 59, 37, 61, 75, 65, 43],
    [50, 54,  2, 45, 55, 67, 79, 69, 77],
    [58, 60, 46,  3, 11, 25, 33, 23, 29],
    [40, 38, 56, 12,  4, 47, 21, 17, 35],
    [42, 62, 68, 26, 48,  5, 31, 27, 19],
    [72, 76, 80, 34, 22, 32,  6, 13, 15],
    [74, 66, 70, 24, 18, 28, 14,  7, 63],
    [52, 44, 78, 30, 36, 20, 16, 64,  8]],
    dtype=np.int32)
  elif r_local == "correlation-rms":
    order_table = np.array([
    [ 0, 11, 17, 65, 47, 67, 63, 59, 53],
    [12,  1, 25, 57, 45, 61, 51, 55, 49],
    [18, 26,  2, 77, 69, 79, 75, 73, 71],
    [66, 58, 78,  3, 13, 21, 43, 31, 33],
    [48, 46, 70, 14,  4, 41, 39, 37, 35],
    [68, 62, 80, 22, 42,  5, 29, 27, 19],
    [64, 52, 76, 44, 40, 30,  6, 15,  9],
    [60, 56, 74, 32, 38, 28, 16,  7, 23],
    [54, 50, 72, 34, 36, 20, 10, 24,  8]],
    dtype=np.int32)
  elif r_local == "covariance-mean":
    order_table = np.array([
    [ 9,  6, 35, 75, 49, 69, 59, 43, 55],
    [ 7,  2, 28, 73, 47, 65, 57, 45, 51],
    [36, 29,  3, 63, 61, 71, 77, 79, 53],
    [76, 74, 64, 34, 32, 39, 41, 30, 22],
    [50, 48, 62, 33, 21, 67, 37, 15, 19],
    [70, 66, 72, 40, 68, 14, 26, 10, 12],
    [60, 58, 78, 42, 38, 27,  8,  4, 24],
    [44, 46, 80, 31, 16, 11,  5,  0, 17],
    [56, 52, 54, 23, 20, 13, 25, 18,  1]],
    dtype=np.int32)
  elif r_local == "covariance-rms":
    order_table = np.array([
    [21, 10, 14, 79, 75, 69, 71, 53, 55],
    [11,  4, 12, 73, 63, 57, 61, 47, 51],
    [15, 13,  7, 77, 67, 65, 59, 43, 49],
    [80, 74, 78, 42, 40, 38, 45, 29, 27],
    [76, 64, 68, 41, 33, 34, 36, 22, 24],
    [70, 58, 66, 39, 35, 26, 31, 18, 16],
    [72, 62, 60, 46, 37, 32, 20,  5,  8],
    [54, 48, 44, 30, 23, 19,  6,  1,  2],
    [56, 52, 50, 28, 25, 17,  9,  3,  0]],
    dtype=np.int32)
  elif r_local == "bhhtri-mean":
    order_table = np.array([
    [ 4,  2, 21, 67, 43, 61, 77, 71, 75],
    [ 3,  0, 15, 65, 37, 57, 76, 72, 73],
    [22, 16,  1, 55, 53, 63, 78, 80, 74],
    [68, 66, 56, 20, 18, 26, 69, 47, 40],
    [44, 38, 54, 19, 10, 59, 49, 32, 39],
    [62, 58, 64, 27, 60,  7, 46, 30, 31],
    [52, 51, 70, 28, 23, 14, 29, 24, 41],
    [33, 34, 79, 17,  8,  5, 25, 12, 35],
    [50, 45, 48, 11,  9,  6, 42, 36, 13]],
    dtype=np.int32)
  elif r_local == "bhhtri-rms":
    order_table = np.array([
    [10,  2,  6, 67, 61, 56, 80, 76, 77],
    [ 3,  0,  4, 59, 48, 41, 79, 73, 75],
    [ 7,  5,  1, 64, 52, 50, 78, 71, 74],
    [68, 60, 65, 29, 27, 25, 72, 66, 63],
    [62, 49, 53, 28, 19, 20, 70, 54, 55],
    [57, 42, 51, 26, 21, 13, 69, 46, 43],
    [58, 45, 44, 31, 24, 18, 47, 33, 35],
    [39, 32, 30, 15, 11,  9, 34, 17, 22],
    [40, 38, 37, 14, 12,  8, 36, 23, 16]],
    dtype=np.int32)
  elif r_local == "random":
    order_table = np.array([
    [68, 54, 40, 11, 48, 79, 66, 34, 73],
    [53, 43, 14, 42, 26, 13,  1, 35, 64],
    [52, 21, 28, 27, 23, 44, 62, 16, 70],
    [74,  5,  0, 10,  2,  7, 60, 65,  6],
    [19, 37, 49, 69, 30, 45,  4, 63, 78],
    [20, 18, 51, 38, 32, 72, 33,  9, 59],
    [22, 55, 41, 39, 29, 61, 57,  8, 58],
    [67, 75, 77, 76, 71, 47, 36, 50, 56],
    [12, 46,  3, 31, 80, 15, 24, 25, 17]],
    dtype=np.int32)
  return order_table

