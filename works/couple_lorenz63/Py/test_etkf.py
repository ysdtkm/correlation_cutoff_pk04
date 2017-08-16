from unittest import TestCase
import etkf
from const import geth, getr, N_MODEL, P_OBS
import numpy as np


class TestEtkf(TestCase):
    def test_etkf_reduced_spread(self):
        def spread(ens_x):
            mean_x = np.mean(ens_x, axis=0)
            ptb_x = ens_x - mean_x[np.newaxis, :]
            sprd = np.sum(ptb_x ** 2, axis=0) / (nmem - 1.0)
            return sprd

        h = geth()
        r = getr()
        for nmem in range(2, 10):
            for i in range(10):
                fcst = np.random.randn(nmem, N_MODEL)
                yo = np.random.randn(P_OBS, 1)
                rho = 1.0

                anl, dummy, dummy, dummy = etkf.etkf(fcst, h, r, yo, rho, nmem, None, True, "full", 0)
                anl_spread = spread(anl)
                fcst_spread = spread(fcst)
                self.assertTrue((anl_spread < fcst_spread).all())

    def test_etkf_reduced_err(self):
        def rmse(true_x, ens_x):
            mean_x = np.mean(ens_x, axis=0)
            abserr = np.abs(true_x - mean_x)
            return (abserr ** 2).sum() ** 0.5

        h = geth()
        r = getr()

        for nmem in range(2, 10):
            anl_wins_fcst = []
            for i in range(100):
                fcst = np.random.randn(nmem, N_MODEL)
                true = np.random.randn(N_MODEL)
                obs = h.dot(true[:])  # no obs error
                yo = obs[:, np.newaxis]
                rho = 1.0

                anl, dummy, dummy, dummy = etkf.etkf(fcst, h, r, yo, rho, nmem, None, True, "full", 0)
                anl_err = rmse(true, anl)
                fcst_err = rmse(true, fcst)
                anl_wins_fcst.append(anl_err < fcst_err)

            wins = anl_wins_fcst.count(True)
            loses = anl_wins_fcst.count(False)
            print("wins:", wins, "loses:", loses)
            self.assertTrue(wins > loses)
