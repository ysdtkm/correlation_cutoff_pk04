#!/usr/bin/env python


import stats_const
import numpy as np
from scipy.optimize import fmin_bfgs


def tdvar(fcst: np.ndarray, h: np.ndarray, r: np.ndarray, yo: np.ndarray,
          i_s: int, i_e: int, amp_b: float) -> np.ndarray:
    """
    :param fcst:  [dimc] first guess
    :param h:     [pc_obs, dimc] observation operator
    :param r:     [pc_obs, pc_obs] observation error covariance
    :param yo:    [pc_obs, 1] observation
    :param i_s:
    :param i_e:
    :param amp_b:
    :return anl:  [dimc] assimilated field
    """
    anl = np.copy(fcst)
    anl = fmin_bfgs(tdvar_2j, anl, args=(fcst, h, r, yo, i_s, i_e, amp_b), disp=False)
    return anl.flatten()


def tdvar_2j(anl_nda: np.ndarray, fcst_nda: np.ndarray, h_nda: np.ndarray, r_nda: np.ndarray,
             yo_nda: np.ndarray, i_s: int, i_e: int, amp_b: float) -> np.ndarray:
    """
    :param anl_nda:  [dimc] temporary analysis field
    :param fcst_nda: [dimc] first guess field
    :param h_nda:    [pc_obs, dimc] observation operator
    :param r_nda:    [pc_obs, pc_obs] observation error covariance
    :param yo_nda:   [pc_obs, 1] observation
    :param i_s:      first model grid to assimilate
    :param i_e:      last model grid to assimilate
    :param amp_b:
    :return:         cost function 2J
    """
    h = np.asmatrix(h_nda)
    r = np.asmatrix(r_nda)
    yo = np.asmatrix(yo_nda)
    b = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])

    anl = np.asmatrix(anl_nda).T
    fcst = np.asmatrix(fcst_nda).T
    twoj = (anl - fcst).T * b.I * (anl - fcst) + \
           (h * anl - yo).T * r.I * (h * anl - yo)
    return twoj[0, 0]


def tdvar_interpol(fcst: np.ndarray, h_nda: np.ndarray, r_nda: np.ndarray, yo_nda: np.ndarray,
                   i_s: int, i_e: int, amp_b: float) -> np.ndarray:
    """
    Return same analysis with tdvar(), not by minimization but analytical interpolation

    :param fcst:    [dimc] first guess
    :param h_nda:   [pc_obs, dimc] observation operator
    :param r_nda:   [pc_obs, pc_obs] observation error covariance
    :param yo_nda:  [pc_obs, 1] observation
    :param i_s:
    :param i_e:
    :param amp_b:
    :return anl:    [dimc] assimilated field
    """
    xb = np.asmatrix(fcst).T
    h = np.asmatrix(h_nda)
    r = np.asmatrix(r_nda)
    yo = np.asmatrix(yo_nda)
    b = np.matrix(amp_b * stats_const.tdvar_b()[i_s:i_e, i_s:i_e])

    d = yo - h * xb

    inc_model = (b.I + h.T * r.I * h).I * h.T * r.I * d
    anl = (xb + inc_model).A.flatten()
    return anl
