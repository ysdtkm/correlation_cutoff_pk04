#!/usr/bin/env python

import os
import numpy as np
from const import N_MODEL, STEPS, AINT, DT, FERR_INI
import model
import main


def obtain_climatology():
    nstep = 100000
    all_true = np.empty((nstep, N_MODEL))

    np.random.seed((10 ** 8 + 7) * 11)
    true = np.random.randn(N_MODEL) * FERR_INI

    for i in range(0, nstep):
        true[:] = model.timestep(true[:], DT)
        all_true[i, :] = true[:]
    # all_true.tofile("data/true_for_clim.bin")

    mean = np.mean(all_true[nstep // 2:, :], axis=0)
    print("mean")
    print(mean)

    mean2 = np.mean(all_true[nstep // 2:, :] ** 2, axis=0)
    stdv = np.sqrt(mean2 - mean ** 2)
    print("stdv")
    print(stdv)

    return 0


def obtain_tdvar_b():
    np.random.seed((10 ** 8 + 7) * 12)
    nature = main.exec_nature()
    obs = main.exec_obs(nature)
    settings = {"name": "etkf_strong_int8", "rho": 1.1, "aint": 8, "nmem": 10,
                "method": "etkf", "couple": "strong", "r_local": "full"}
    np.random.seed((10 ** 8 + 7) * 13)
    free = main.exec_free_run(settings)
    anl = main.exec_assim_cycle(settings, free, obs)
    hist_bf = np.fromfile("data/%s_covr_back.bin" % settings["name"], np.float64)
    hist_bf = hist_bf.reshape((STEPS, N_MODEL, N_MODEL))
    mean_bf = np.nanmean(hist_bf[STEPS // 2:, :, :], axis=0)
    trace = np.trace(mean_bf)
    mean_bf *= (N_MODEL / trace)

    print("[ \\")
    for i in range(N_MODEL):
        print("[", end="")
        for j in range(N_MODEL):
            print("%12.9g" % mean_bf[i, j], end="")
            if j < (N_MODEL - 1):
                print(", ", end="")
        if i < (N_MODEL - 1):
            print("], \\")
        else:
            print("]  \\")
            print("]")

    return 0


def print_two_dim_nparray(data, fmt="%12.9g"):
    n = data.shape[0]
    m = data.shape[1]
    print("[ \\")
    for i in range(n):
        print("[", end="")
        for j in range(m):
            print(fmt % data[i, j], end="")
            if j < (m - 1):
                print(", ", end="")
        if i < (n - 1):
            print("], \\")
        else:
            print("]  \\")
            print("]")


def obtain_stats_etkf():
    """
    For lagged correlation, delta_t is always defined as tj - ti.
    (i.e., delta_t > 0 iff x_j is defined later than x_i)
    """
    def cov_to_corr(cov):
        corr = cov.copy()
        for i in range(N_MODEL):
            for j in range(N_MODEL):
                corr[i, j] /= np.sqrt(cov[i, i] * cov[j, j])
        return corr

    def obtain_cycle():
        np.random.seed((10 ** 8 + 7) * 12)
        nature = main.exec_nature()
        obs = main.exec_obs(nature)
        settings = dict(name="etkf", rho="adaptive", nmem=10, method="etkf", couple="strong", r_local="full")
        np.random.seed((10 ** 8 + 7) * 13)
        free = main.exec_free_run(settings)
        anl = main.exec_assim_cycle(settings, free, obs)

        nmem = settings["nmem"]
        hist_fcst = np.fromfile("data/%s_cycle.bin" % settings["name"], np.float64)
        hist_fcst = hist_fcst.reshape((STEPS, nmem, N_MODEL))
        return hist_fcst, nature, nmem

    def obtain_instant_covs_corrs(hist_fcst, nmem, delta_ti, delta_tj):
        # a40 (17.3)
        corr_ijt = np.empty((STEPS, N_MODEL, N_MODEL))
        corr_ijt[:, :, :] = np.nan
        cov_ijt = np.empty((STEPS, N_MODEL, N_MODEL))
        cov_ijt[:, :, :] = np.nan

        for it in range(STEPS // 2, STEPS):
            if it % AINT == 0:
                fcsti = hist_fcst[it, :, :].copy()
                fcstj = hist_fcst[it, :, :].copy()
                for k in range(nmem):
                    for jt in range(delta_ti):
                        fcsti[k, :] = model.timestep(fcsti[k, :], DT)
                    for jt in range(delta_tj):
                        fcstj[k, :] = model.timestep(fcstj[k, :], DT)

                for i in range(N_MODEL):
                    for j in range(N_MODEL):
                        # a38p40
                        vector_i = np.copy(fcsti[:, i])
                        vector_j = np.copy(fcstj[:, j])
                        vector_i[:] -= np.mean(vector_i)
                        vector_j[:] -= np.mean(vector_j)
                        numera = np.sum(vector_i * vector_j)
                        denomi = (np.sum(vector_i ** 2) * np.sum(vector_j ** 2)) ** 0.5
                        corr_ijt[it, i, j] = numera / denomi
                        cov_ijt[it, i, j] = numera / (nmem - 1.0)
        return corr_ijt, cov_ijt

    def obtain_clim_covs_corrs(nature, delta_ti, delta_tj):
        # a40 (17.4)

        cov_clim_ij = np.zeros((N_MODEL, N_MODEL))
        var_clim_i = np.zeros(N_MODEL)
        var_clim_j = np.zeros(N_MODEL)
        k = 0
        step_start = STEPS // 2
        step_end = STEPS - max(delta_ti, delta_tj)
        mean1 = np.mean(nature[step_start+delta_ti:step_end+delta_ti, :], axis=0)
        mean2 = np.mean(nature[step_start+delta_tj:step_end+delta_tj, :], axis=0)
        for it in range(step_start, step_end):
            anom1 = nature[it + delta_ti, :] - mean1[:]
            anom2 = nature[it + delta_tj, :] - mean2[:]
            for i in range(N_MODEL):
                cov_clim_ij[i, :] += anom1[i] * anom2[:]
            var_clim_i[:] += anom1[:] ** 2
            var_clim_j[:] += anom2[:] ** 2
            k += 1

        corr_clim_ij = cov_clim_ij.copy()
        for i in range(N_MODEL):
            corr_clim_ij[i, :] /= var_clim_j[:] ** 0.5
        for j in range(N_MODEL):
            corr_clim_ij[:, j] /= var_clim_i[:] ** 0.5

        cov_clim_ij[:, :] /= (k - 1.0)

        return corr_clim_ij, cov_clim_ij

    def reduce_covs_corrs(corr_ijt, cov_ijt):
        corr_mean_ij = np.nanmean(corr_ijt, axis=0)
        corr_rms_ij = np.sqrt(np.nanmean(corr_ijt ** 2, axis=0))
        cov_mean_ij = np.nanmean(cov_ijt, axis=0)
        cov_rms_ij = np.sqrt(np.nanmean(cov_ijt ** 2, axis=0))
        return corr_mean_ij, corr_rms_ij, cov_mean_ij, cov_rms_ij

    def save_data(mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij,
                  clim_corr_ij, clim_cov_ij, num_delta_t, delta_t_set):
        datadir = "data/offline"
        os.system("mkdir -p %s" % datadir)
        name_and_data = dict(mean_corr = mean_corr_ij, rms_corr = rms_corr_ij,
                             mean_cov = mean_cov_ij, rms_cov = rms_cov_ij,
                             clim_cov = clim_cov_ij, clim_corr = clim_corr_ij,
                             num_delta_t = np.array(num_delta_t), delta_t_set = np.array(delta_t_set))
        for name in name_and_data:
            data = name_and_data[name]
            np.save("%s/%s.npy" % (datadir, name), data)

    os.system("mkdir -p data")
    print("calculating analysis cycle")
    hist_fcst, nature, nmem = obtain_cycle()

    print("calculating statistics")
    num_delta_t = 26
    delta_t_set = list(np.linspace(0, 50, num_delta_t, dtype=np.int))

    # delta_t, i, j
    mean_corr_ij = np.empty((num_delta_t, N_MODEL, N_MODEL))
    rms_corr_ij = np.empty((num_delta_t, N_MODEL, N_MODEL))
    mean_cov_ij = np.empty((num_delta_t, N_MODEL, N_MODEL))
    rms_cov_ij = np.empty((num_delta_t, N_MODEL, N_MODEL))
    clim_cov_ij = np.empty((num_delta_t, N_MODEL, N_MODEL))
    clim_corr_ij = np.empty((num_delta_t, N_MODEL, N_MODEL))

    for it, delta_t in enumerate(delta_t_set):
        delta_ti = 0
        delta_tj = delta_t
        corr_ijt, cov_ijt = obtain_instant_covs_corrs(hist_fcst, nmem, delta_ti, delta_tj)
        clim_corr_ij[it, :, :], clim_cov_ij[it, :, :] = obtain_clim_covs_corrs(nature, delta_ti, delta_tj)
        mean_corr_ij[it, :, :], rms_corr_ij[it, :, :], mean_cov_ij[it, :, :], rms_cov_ij[it, :, :] = \
            reduce_covs_corrs(corr_ijt, cov_ijt)

    save_data(mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij,
              clim_corr_ij, clim_cov_ij, num_delta_t, delta_t_set)


if __name__ == "__main__":
    obtain_stats_etkf()
