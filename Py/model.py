#!/usr/bin/env python

import numpy as np
from const import N_MODEL


def timestep(x: np.ndarray, dt: float, i_s: int = 0, i_e: int = N_MODEL, bc: np.ndarray = None) -> np.ndarray:
    """
    :param x:   [dimm]
    :param dt:
    :param i_s:
    :param i_e:
    :param bc:  [N_MODEL] boundary condition
    :return x:  [dimm]
    """

    x0 = np.copy(x)
    k1 = tendency(x0, i_s, i_e, bc)
    x2 = x0 + k1 * dt / 2.0
    k2 = tendency(x2, i_s, i_e, bc)
    x3 = x0 + k2 * dt / 2.0
    k3 = tendency(x3, i_s, i_e, bc)
    x4 = x0 + k3 * dt
    k4 = tendency(x4, i_s, i_e, bc)
    return x0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0


def tendency(x_in: np.ndarray, i_s: int = 0, i_e: int = N_MODEL, bc: np.ndarray = None) -> np.ndarray:
    """
    (dimm = i_e - i_s <= N_MODEL) unless strongly coupled, a31p63-64

    :param x_in: [dimm]
    :param i_s:
    :param i_e:
    :param bc:   [N_MODEL]
    :return dx:  [dimm]
    """
    if N_MODEL == 3:
        sigma = 10.0
        r = 28.0
        b = 8.0 / 3.0
        k = np.empty(3)
        k[0] = -sigma * x_in[0] + sigma * x_in[1]
        k[1] = -x_in[0] * x_in[2] + r * x_in[0] - x_in[1]
        k[2] = x_in[0] * x_in[1] - b * x_in[2]
        return k

    elif N_MODEL == 9:
        # model constants
        sigma = 10.0
        r = 28.0
        b = 8.0 / 3.0
        tau = 0.1
        c = 1.0
        cz = 1.0
        ce = 0.08
        s = 1.0
        k1 = 10.0
        k2 = -11.0

        # set boundary conditions
        if i_s == 0 and i_e == N_MODEL:
            x = np.copy(x_in)
        else:
            if bc is None:
                # climatology B.C. for non-coupled. Obtained from (a757b4e) unit_test.py
                x = np.array([0.35128345, 0.41208204, 23.57932048, -2.68240888, -2.26472921,
                              29.22828843, 14.33420545, 0.65398139, 16.64817181])
            else:
                x = np.copy(bc)
            x[i_s:i_e] = x_in[:]

        dx = np.empty(N_MODEL)
        # extratropic atm
        dx[0] = -sigma * x[0] + sigma * x[1] - ce * (s * x[3] + k1)
        dx[1] = -x[0] * x[2] + r * x[0] - x[1] + ce * (s * x[4] + k1)
        dx[2] = x[0] * x[1] - b * x[2]
        # tropic atm
        dx[3] = -sigma * x[3] + sigma * x[4] - c * (s * x[6] + k2) - ce * (s * x[0] + k1)
        dx[4] = -x[3] * x[5] + r * x[3] - x[4] + c * (s * x[7] + k2) + ce * (s * x[1] + k1)
        dx[5] = x[3] * x[4] - b * x[5] + cz * x[8]
        # tropic ocean
        dx[6] = tau * (-sigma * x[6] + sigma * x[7]) - c * (x[3] + k2)
        dx[7] = tau * (-s * x[6] * x[8] + r * x[6] - x[7]) + c * (x[4] + k2)
        dx[8] = tau * (s * x[6] * x[7] - b * x[8]) - cz * x[5]
        return dx[i_s:i_e]

    else:
        f = 8.0
        k = (np.roll(x_in, -1) - np.roll(x_in, 2)) * np.roll(x_in, 1) - x_in + f
        return k


def tangent_linear(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Return one-timestep tangent linear matrix, currently not for non/weakly coupled
    Uses Eular-forward. Similar to Jacobian.

    :param x:  [N_MODEL] state vector at the beginning
    :param dt: infinitesimal time
    :return:   [N_MODEL,N_MODEL] instantaneous tangent linear matrix M
    """

    dx = np.zeros((N_MODEL, N_MODEL))

    if N_MODEL == 3:
        sigma = 10.0
        r = 28.0
        b = 8.0 / 3.0

        dx[0, 0] = -sigma
        dx[0, 1] = sigma
        dx[0, 2] = 0.0

        dx[1, 0] = -x[2] + r
        dx[1, 1] = -1.0
        dx[1, 2] = -x[0]

        dx[2, 0] = x[1]
        dx[2, 1] = x[0]
        dx[2, 2] = -b

    elif N_MODEL == 9:
        sigma = 10.0
        r = 28.0
        b = 8.0 / 3.0
        tau = 0.1
        c = 1.0
        cz = 1.0
        ce = 0.08
        s = 1.0
        # these parameters are not used because they are constants
        # k1 = 10.0
        # k2 = -11.0

        # extratropic atm
        dx[0, 0] = -sigma
        dx[0, 1] = sigma
        dx[0, 3] = -ce * s

        dx[1, 0] = -x[2] + r
        dx[1, 1] = -1.0
        dx[1, 2] = -x[0]
        dx[1, 4] = ce * s

        dx[2, 0] = x[1]
        dx[2, 1] = x[0]
        dx[2, 2] = -b

        # tropic atm
        dx[3, 0] = -ce * s
        dx[3, 3] = -sigma
        dx[3, 4] = sigma
        dx[3, 6] = -c * s

        dx[4, 1] = ce * s
        dx[4, 3] = -x[5] + r
        dx[4, 4] = -1.0
        dx[4, 5] = -x[3]
        dx[4, 7] = c * s

        dx[5, 3] = x[4]
        dx[5, 4] = x[3]
        dx[5, 5] = -b
        dx[5, 8] = cz

        # tropic ocean
        dx[6, 3] = - c
        dx[6, 6] = tau * (-sigma)
        dx[6, 7] = tau * sigma

        dx[7, 4] = c
        dx[7, 6] = tau * (-s * x[8] + r)
        dx[7, 7] = -tau
        dx[7, 8] = tau * (-s * x[6])

        dx[8, 5] = -cz
        dx[8, 6] = tau * (s * x[7])
        dx[8, 7] = tau * (s * x[6])
        dx[8, 8] = tau * (-b)

    m = np.identity(N_MODEL) + dx[:, :] * dt
    return m


def finite_time_tangent(x0: np.ndarray, dt: float, iw: int) -> np.ndarray:
    """
    Return finite time tangent linear matrix (t0 -> t0 + dt * iw)

    :param x0: [N_MODEL] state vector at the beginning of the window
    :param dt: length of a time step
    :param iw: integration window (time in steps)
    :return:   [N_MODEL,N_MODEL]
    """

    m_finite = np.identity(N_MODEL)
    x = np.copy(x0)
    for i in range(iw):
        m_inst = tangent_linear(x, dt)
        m_finite = np.dot(m_inst, m_finite)
        x = timestep(x, dt)
    return m_finite


def finite_time_tangent_using_nonlinear(x0: np.ndarray, dt: float, iw: int) -> np.ndarray:
    """
    Return tangent linear matrix, calculated numerically using the NL model
    todo: boundary conditions needed if used for 4DVar

    :param x0: [N_MODEL] state vector at the beginning of the window
    :param dt: length of a time step
    :param iw: integration window (time in steps)
    :return:   [N_MODEL,N_MODEL]
    """

    m_finite = np.identity(N_MODEL)
    eps = 1.0e-9
    for j in range(N_MODEL):
        xctl = np.copy(x0)
        xptb = np.copy(x0)
        xptb[j] += eps
        for i in range(iw):
            xctl = timestep(xctl, dt)
            xptb = timestep(xptb, dt)
        m_finite[:, j] = (xptb[:] - xctl[:]) / eps
    return m_finite
