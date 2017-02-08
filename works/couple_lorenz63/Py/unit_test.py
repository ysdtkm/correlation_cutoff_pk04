#!/usr/bin/env python

import numpy as np
from const import *
from fdvar import *

def test_fdvar_overflow():
  exp = EXPLIST[1]
  fcst_0 = np.array([[ -5.44300627], [ -0.80439722], [ 29.44835342], \
                     [ -6.8756489 ], [-10.74616235], [ 45.01895925], \
                     [-40.33122968], [-92.74048173], [ 81.47469208]])
  h = geth(exp["diag"])
  r = getr()
  yo = np.matrix([[  -2.93129428], [  -5.97242816], [  14.37325925], \
                  [   0.14974816], [  -3.39355898], [  47.92023891], \
                  [ -39.34468967], [ -12.51275803], [ 117.90228464]])
  aint = exp["aint"]

  fdvar(fcst_0, h, r, yo, aint)

test_fdvar_overflow()
