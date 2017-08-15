#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import N_MODEL, P_OBS, ETKF_AI_max, ETKF_AI_min, ETKF_kappa, ETKF_vo
import stats_const


def etkf(fcst: np.ndarray, h_nda: np.ndarray, r_nda: np.ndarray, yo_nda: np.ndarray, rho_in: float, nmem: int,
         obj_adaptive: object, localization: bool = False, r_local: str = "", num_yes: int = None) -> tuple:
    """
    :param fcst:         [nmem, dimc]
    :param h_nda:        [pc_obs, dimc]
    :param r_nda:        [pc_obs, pc_obs]
    :param yo_nda:       [pc_obs, 1]
    :param rho_in:
    :param nmem:
    :param obj_adaptive: object created by etkf/init_etkf_adaptive_inflation(), or None
    :param localization:
    :param r_local:      localization pattern name of R
    :param num_yes:
    :return xa:          [dimc, nmem]
    :return xfpt:        [dimc, dimc]
    :return xapt:        [dimc, dimc]
    :return obj_adaptive:
    """

    h = np.asmatrix(h_nda)
    r = np.asmatrix(r_nda)
    yo = np.asmatrix(yo_nda)
    dimc = fcst.shape[1]
    pc_obs = yo_nda.shape[0]

    # DA variables (np.matrix)
    # - model space
    #    xfm[dimc,nmem] : each member's forecast Xf
    #   xfpt[dimc,nmem] : forecast perturbation Xbp
    #    xam[dimc,nmem] : each member's analysis Xa
    #   xapt[dimc,nmem] : analysis perturbation Xap
    #     xf[dimc,1   ] : ensemble mean forecast xf_bar
    #     xa[dimc,1   ] : ensemble mean analysis xa_bar
    #
    # - obs space
    #   r[pc_obs,pc_obs] : obs error covariance matrix R
    #     h[pc_obs,dimc] : linearized observation operator H
    #    yo[pc_obs,1   ] : observed state yo
    #    yb[pc_obs,1   ] : mean simulated observation vector yb
    #   ybm[pc_obs,nmem] : ensemble model simulated observation matrix Yb
    #
    # - mem space
    #   wam[nmem,nmem] : optimal weight matrix for each member
    #    wa[nmem,1   ] : optimal weight for each member
    #    pa[nmem,nmem] : approx. anl error covariance matrix Pa in ens space

    i_mm = np.matrix(np.identity(nmem))
    i_1m = np.matrix(np.ones((1, nmem)))

    xfm = np.matrix(fcst[:, :]).T
    xf = np.mean(xfm, axis=1)
    xfpt = xfm - xf * i_1m
    ybpt = h * xfpt
    yb = h * xf

    if localization:
        xai = np.matrix(np.zeros((dimc, nmem)))

        if rho_in == "adaptive":
            delta_this = obtain_delta_for_adaptive_inflation(yo, yb, ybpt, r, nmem, True)
            obj_adaptive = update_adaptive_inflation(obj_adaptive, delta_this)
        elif rho_in == "adaptive_each":
            delta_this = obtain_delta_for_adaptive_inflation(yo, yb, ybpt, r, nmem, False)
            obj_adaptive = update_adaptive_inflation(obj_adaptive, delta_this)

        for j in range(dimc):
            # step 3
            localization_weight = obtain_localization_weight(pc_obs, j, r_local, num_yes)
            yol = yo[:, :].copy()
            ybl = yb[:, :].copy()
            ybptl = ybpt[:, :].copy()
            xfl = xf[j, :].copy()
            xfptl = xfpt[j, :].copy()
            rl = r[:, :].copy()

            if rho_in == "adaptive" or rho_in == "adaptive_each":
                component = [0, 0, 0, 1, 1, 1, 2, 2, 2]
                rho = obj_adaptive[0, component[j]]
            else:
                rho = rho_in

            # step 4-9
            cl = ybptl.T * np.asmatrix(rl.I.A * localization_weight.A)
            pal = (((nmem - 1.0) / rho) * i_mm + cl * ybptl).I
            waptl = np.matrix(np.real(sqrtm((nmem - 1.0) * pal)))
            wal = pal * cl * (yol - ybl)
            xail = xfl * i_1m + xfptl * (wal * i_1m + waptl)
            xai[j, :] = xail[:, :]

        xapt = xai - np.mean(xai[:, :], axis=1) * i_1m
        return np.real(xai.T.A), (xfpt * xfpt.T).A, (xapt * xapt.T).A, obj_adaptive

    else:

        if rho_in == "adaptive" or rho_in == "adaptive_each":
            raise Exception("non-localized ETKF cannot handle adaptive inflation")
        else:
            rho = rho_in

        pa = (((nmem - 1.0) / rho) * i_mm + ybpt.T * r.I * ybpt).I
        wam = np.matrix(sqrtm((nmem - 1.0) * pa))
        wa = pa * ybpt.T * r.I * (yo - yb)
        xapt = (xfm - xf * i_1m) * wam
        xa = xf + xfm * wa
        xam = xapt + xa * i_1m
        return np.real(xam.T.A), (xfpt * xfpt.T).A, (xapt * xapt.T).A, obj_adaptive


def obtain_localization_weight(pc_obs: int, j: int, r_local: str, num_yes: int) -> np.matrix:
    """
    :param pc_obs:  number of obs
    :param j:       index of analyzed grid
    :param r_local: localization pattern of R
    :param num_yes:
    :return:        [pc_obs,pc_obs] localizaiton weight matrix for R-inverse
    """

    def _get_weight_table(r_local_name, num_yes_int):
        # return weight_table[iy, ix] : weight of iy-th obs for ix-th grid

        if r_local_name in ["covariance-mean", "correlation-mean", "covariance-rms",
                            "correlation-rms", "random", "BHHtRi-mean", "BHHtRi-rms",
                            "covariance-clim", "correlation-clim"]:
            if num_yes_int is None:
                num_yes_int = 37
            weight_table_stat = _weight_based_on_stats(r_local_name, num_yes_int)
        elif r_local_name == "dynamical":  # a38p35
            weight_table_stat = np.array([
                [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1]],
                dtype=np.float64)
        else:
            weight_table_small = {
                "individual":       np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                "atmos_coupling":   np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                "enso_coupling":    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
                "atmos_sees_ocean": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
                "trop_sees_ocean":  np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
                "ocean_sees_atmos": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
                "ocean_sees_trop":  np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
                "full":             np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                "adjacent":         np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            }
            weight_table_stat = np.ones((N_MODEL, N_MODEL))
            for iyc in range(3):
                for ixc in range(3):
                    weight_table_stat[iyc * 3:iyc * 3 + 3, ixc * 3:ixc * 3 + 3] = \
                        weight_table_small[r_local_name][iyc, ixc]
        return weight_table_stat

    def _weight_based_on_stats(r_local_name, odr_max=37):
        order_table = stats_const.stats_order(r_local_name)
        return np.float64(order_table < odr_max)

    if not (P_OBS == N_MODEL == 9):
        import warnings
        warnings.warn("obtain_localization_weight() is only for PK04 and P_OBS == N_MODEL. No localization is done.")

        localization_weight = np.ones((pc_obs, pc_obs))
        return np.asmatrix(localization_weight)
    else:
        localization_weight = np.ones((pc_obs, pc_obs))
        if pc_obs == N_MODEL:  # strongly coupled, full-ranked obs
            weight_table = _get_weight_table(r_local, num_yes)
            for iy in range(pc_obs):
                localization_weight[iy, :] *= weight_table[iy, j]
                localization_weight[:, iy] *= weight_table[iy, j]
        elif pc_obs == 3:
            pass
        elif pc_obs == 6:
            pass
        return np.asmatrix(localization_weight)


def init_etkf_adaptive_inflation() -> np.ndarray:
    """
    :return: ((delta_extra, delta_trop, delta_ocean), (var_extra, var_trop, var_ocean))
    """
    if N_MODEL != 9 or P_OBS != N_MODEL:
        raise Exception("Adaptive inflation is only for N_MODEL == P_OBS == 9")

    obj_adaptive = np.array([[1.05, 1.05, 1.05],
                             [1.0, 1.0, 1.0]])
    return obj_adaptive


def update_adaptive_inflation(obj_adaptive: np.ndarray, delta_this_step: np.ndarray) -> np.ndarray:
    """
    :param obj_adaptive:    ((delta_extra, delta_trop, delta_ocean), (var_extra, var_trop, var_ocean))
    :param delta_this_step: (delta_extra, delta_trop, delta_ocean)
    :return:                ((delta_extra, delta_trop, delta_ocean), (var_extra, var_trop, var_ocean))
    """
    if N_MODEL != 9 or P_OBS != N_MODEL:
        raise Exception("Adaptive inflation is only for N_MODEL == P_OBS == 9")

    # limit delta_this_step
    delta_max = np.ones(3) * ETKF_AI_max
    delta_min = np.ones(3) * ETKF_AI_min
    delta_this_step = np.max(np.row_stack((delta_min, delta_this_step)), axis=0)
    delta_this_step = np.min(np.row_stack((delta_max, delta_this_step)), axis=0)

    vf = obj_adaptive[1, :].copy() * ETKF_kappa
    delta_new = (obj_adaptive[0, :] * ETKF_vo + delta_this_step[:] * vf[:]) / (ETKF_vo + vf[:])
    va = (vf[:] * ETKF_vo) / (vf[:] + ETKF_vo)

    obj_adaptive[0, :] = delta_new[:]
    obj_adaptive[1, :] = va[:]

    return obj_adaptive


def obtain_delta_for_adaptive_inflation(yo: np.ndarray, yb: np.ndarray, ybpt: np.ndarray, r: np.ndarray,
                                        nmem: int, common: bool) -> np.ndarray:
    """
    :param yo:     [P_OBS]
    :param yb:     [P_OBS]
    :param ybpt:   [P_OBS, nmem]
    :param r:      [P_OBS, P_OBS]
    :param nmem:
    :param common:
    :return:       (delta_extra, delta_trop, delta_ocean)
    """
    if N_MODEL != 9 or P_OBS != N_MODEL or yo.shape[0] != 9:
        raise Exception("Adaptive inflation is only for N_MODEL == P_OBS == 9")

    delta = np.empty(3)

    if common:
        dob = yo - yb
        delta[:] = (dob.T.dot(dob) - np.trace(r)) / np.trace(ybpt.dot(ybpt.T) / (nmem - 1))
    else:
        for i in range(3):
            i_s = i * 3
            i_e = i * 3 + 3
            yol = yo[i_s:i_e]
            ybl = yb[i_s:i_e]
            ybptl = ybpt[i_s:i_e, :]
            rl = r[i_s:i_e, i_s:i_e]
            dob = yol - ybl
            delta[i] = (dob.T.dot(dob) - np.trace(rl)) / np.trace(ybptl.dot(ybptl.T) / (nmem - 1))
    return delta
