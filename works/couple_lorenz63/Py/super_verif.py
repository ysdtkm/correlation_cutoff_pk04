#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def verif(args):
  wf = open("image/verif.txt", "w")
  wf.write(str(args))
  return
