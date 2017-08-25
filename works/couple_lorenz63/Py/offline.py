#!/usr/bin/env python

import os
import numpy as np
from const import N_MODEL, STEPS, AINT, DT, FERR_INI, getr
import model
import main
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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


def print_two_dim_nparray(data, format="%12.9g"):
    n = data.shape[0]
    m = data.shape[1]
    print("[ \\")
    for i in range(n):
        print("[", end="")
        for j in range(m):
            print(format % data[i, j], end="")
            if j < (m - 1):
                print(", ", end="")
        if i < (n - 1):
            print("], \\")
        else:
            print("]  \\")
            print("]")


def plot_matrix(data, dir, name="", title="", color=plt.cm.bwr, xlabel="", ylabel="", logscale=False, linthresh=1e-3):
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
    cmax = np.max(np.abs(data))
    if logscale:
        map1 = ax.pcolor(data, cmap=color, norm=colors.SymLogNorm(linthresh=linthresh * cmax))
    else:
        map1 = ax.pcolor(data, cmap=color)
    if color == plt.cm.bwr:
        map1.set_clim(-1.0 * cmax, cmax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(map1)
    plt.title(title)
    plt.gca().invert_yaxis()
    os.system("mkdir -p %s" % dir)
    plt.savefig("./%s/matrix_%s_%s.pdf" % (dir, name, title))
    plt.close()
    return 0


def obtain_stats_etkf():
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

    def obtain_covs_corrs(hist_fcst, nature, nmem, delta_ti, delta_tj):

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
                cov_instant_ij = cov_ijt[it, :, :].copy()

        clim_mean = np.mean(nature[STEPS // 2:, :], axis=0)
        cov_clim_ij = np.empty((N_MODEL, N_MODEL))
        k = 0
        for it in range(STEPS // 2, STEPS):
            anom = nature[it, :] - clim_mean[:]
            for i in range(N_MODEL):
                cov_clim_ij[i, :] += anom[i] * anom[:]
            k += 1
        tmp = cov_clim_ij  # to remove assymetry caused by numerical errors
        cov_clim_ij[:, :] += tmp.T
        cov_clim_ij[:, :] /= (2.0 * (k - 1.0))

        return corr_ijt, cov_ijt, cov_instant_ij, cov_clim_ij, diff_t

    def reduce_and_plot(corr_ijt, cov_ijt, cov_instant_ij, cov_clim_ij, diff_t):
        corr_mean_ij = np.nanmean(corr_ijt, axis=0)
        corr_rms_ij = np.sqrt(np.nanmean(corr_ijt ** 2, axis=0))
        cov_mean_ij = np.nanmean(cov_ijt, axis=0)
        cov_rms_ij = np.sqrt(np.nanmean(cov_ijt ** 2, axis=0))

        data_hash = {"correlation-mean": corr_mean_ij, "correlation-rms": corr_rms_ij,
                     "covariance-mean": cov_mean_ij, "covariance-rms": cov_rms_ij}
        if diff_t == 0:
            corr_instant_ij = cov_to_corr(cov_instant_ij)
            corr_clim_ij = cov_to_corr(cov_clim_ij)
            data_hash2 = {"covariance-clim": cov_clim_ij, "correlation-clim": corr_clim_ij,
                          "covariance-instant": cov_instant_ij, "correlation-instant": corr_instant_ij}
            data_hash.update(data_hash2)
        for name in data_hash:
            name2 = name + "_" + str(diff_t)
            dir="offline_%d" % diff_t
            plot_matrix(data_hash[name], dir, title=name2, xlabel="grid index i",
                        ylabel="grid index j", logscale=True, linthresh=1e-1)
            plot_matrix(data_hash[name], dir, title=(name2 + "_linear"), xlabel="grid index i",
                        ylabel="grid index j", logscale=False)
            print(name2)
            try:
                matrix_order(np.abs(data_hash[name]), dir, name2)
            except Exception as e:
                print(e)

    hist_fcst, nature, nmem = obtain_cycle()

    # delta_ti = delta_tj = 0 for analysis, 8 for background correlation
    # delta_ti != delta_tj for lagged correlation
    delt_set = [(0, 0), (0, 8), (0, 16), (0, 24)]
    for delt in delt_set:
        delta_ti = delt[0]
        delta_tj = delt[1]
        diff_t = delta_tj - delta_ti
        corr_ijt, cov_ijt, cov_instant_ij, cov_clim_ij, diff_t = \
            obtain_covs_corrs(hist_fcst, nature, nmem, delta_ti, delta_tj)
        reduce_and_plot(corr_ijt, cov_ijt, cov_instant_ij, cov_clim_ij, diff_t)


def matrix_order(mat_ij_in, dir, name, prioritize_diag=False, max_odr=81):
    n = len(mat_ij_in)
    if len(mat_ij_in[0]) != n:
        raise Exception("input matrix non-square")
    mat_ij = mat_ij_in.copy()

    def find_last_order(sorted_vals, test):
        for i, val in reversed(list(enumerate(sorted_vals))):
            if val == test:
                return i
        raise Exception("find_last_order overflow")

    def print_order(order):
        for i in range(n):
            print("[", end="")
            for j in range(n):
                if order[i][j] < max_odr:
                    print("%2d" % order[i][j], end="")
                else:
                    print("**", end="")
                if j < n - 1:
                    print(", ", end="")
            print("],")
        print("")
        return

    if prioritize_diag:
        for i in range(n):
            mat_ij[i, i] = np.inf

    all_vals = []
    for i in range(n):
        for j in range(n):
            all_vals.append(mat_ij[i, j])
    sorted_vals = sorted(all_vals, reverse=True)

    order = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            order[i][j] = find_last_order(sorted_vals, mat_ij[i, j])

    print_order(order)

    x = np.array(range(1, len(sorted_vals) + 1))
    y = np.array(sorted_vals)
    plt.bar(x, y / np.max(y), label=name)
    plt.xlabel("descending order (1-based)")
    plt.ylabel("relative amplitude")
    plt.yscale("log")
    plt.ylim(1.0e-4, 1.0)
    plt.legend()
    os.system("mkdir -p %s" % dir)
    plt.savefig("%s/histogram_%s.pdf" % (dir, name))
    plt.clf()

    return 0


if __name__ == "__main__":
    obtain_stats_etkf()
