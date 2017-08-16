from unittest import TestCase
import tdvar
import fdvar
from const import geth, getr, N_MODEL, P_OBS, amp_b_tdvar
import numpy as np


class TestTdvar(TestCase):
    def test_equality_tdvar_and_tdvar_interpol(self):
        h = geth()
        r = getr()
        fcst = np.random.randn(N_MODEL)
        obs = np.random.randn(P_OBS)
        yo = obs[:, np.newaxis]

        anl1 = tdvar.tdvar(fcst, h, r, yo, 0, N_MODEL, amp_b_tdvar)
        anl2 = tdvar.tdvar_interpol(fcst, h, r, yo, 0, N_MODEL, amp_b_tdvar)
        eps = 1.0e-5
        self.assertTrue((np.abs(anl1 - anl2) < eps).all())

    def test_equality_tdvar_and_fdvar_zero_window(self):
        h = geth()
        r = getr()
        fcst = np.random.randn(N_MODEL)
        obs = np.random.randn(P_OBS)
        yo = obs[:, np.newaxis]

        anl1 = tdvar.tdvar(fcst, h, r, yo,    0, N_MODEL, amp_b_tdvar)
        anl2 = fdvar.fdvar(fcst, h, r, yo, 0, 0, N_MODEL, amp_b_tdvar)
        anl3 = fdvar.fdvar(fcst, h, r, yo, 8, 0, N_MODEL, amp_b_tdvar)
        eps = 1.0e-9
        self.assertTrue((np.abs(anl1 - anl2) < eps).all())
        self.assertTrue((np.abs(anl2 - anl3) > eps).any())



