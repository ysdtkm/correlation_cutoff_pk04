from unittest import TestCase
import model
from const import N_MODEL, DT, FERR_INI, STEPS
import numpy as np


class TestModel(TestCase):
    def test_timestep_dt_continuity(self):
        """fails if DT is too large"""

        for i in range(100):
            anl = np.random.randn(N_MODEL)
            fcst1 = model.timestep(anl, DT)
            fcst2 = model.timestep(anl, DT / 2)
            fcst2 = model.timestep(fcst2, DT / 2)
            eps = 1.0e-4
            self.assertTrue((np.abs(fcst1 - fcst2) < eps).all())
            self.assertTrue((np.abs(anl - fcst1) > eps).any())

    def test_tangent_model(self):
        """m1 has largest error due to Eular-forward truncation, which is expected to scale O(DT^2)"""

        ptb = 1.0e-6
        eps1 = 0.1
        eps2 = 1.0e-4
        step_verif = 1

        maxd1 = 0.0
        maxd2 = 0.0
        for itr in range(10):
            x_t0 = np.random.randn(N_MODEL) * FERR_INI
            for i in range(STEPS):
                x_t0 = model.timestep(x_t0, DT)
            x_t1 = np.copy(x_t0)
            for i in range(step_verif):
                x_t1 = model.timestep(x_t1, DT)

            m1 = model.finite_time_tangent(x_t0, DT, step_verif)
            m2 = model.finite_time_tangent_using_nonlinear(x_t0, DT, step_verif)
            m3 = np.empty((N_MODEL, N_MODEL))

            for i in range(N_MODEL):
                x_t0_ptb = np.copy(x_t0)
                x_t0_ptb[i] = x_t0[i] + ptb
                x_t1_ptb = np.copy(x_t0_ptb)
                for j in range(step_verif):
                    x_t1_ptb = model.timestep(x_t1_ptb, DT)
                m3[:, i] = (x_t1_ptb - x_t1) / ptb

            maxd1 = max(maxd1, np.max(np.abs(m2 - m1)))
            maxd2 = max(maxd2, np.max(np.abs(m3 - m2)))
            self.assertTrue((np.abs(m2 - m1) < eps1).all())
            self.assertTrue((np.abs(m3 - m2) < eps2).all())
        # print(maxd1, maxd2)


