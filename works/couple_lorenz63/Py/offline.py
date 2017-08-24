#!/usr/bin/env python

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


def plot_matrix(data, name="", title="", color=plt.cm.bwr, xlabel="", ylabel="", logscale=False, linthresh=1e-3):
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
    plt.savefig("./matrix_%s_%s.pdf" % (name, title))
    plt.close()
    return 0


def obtain_stats_etkf():
    def cov_to_corr(cov):
        corr = cov.copy()
        for i in range(N_MODEL):
            for j in range(N_MODEL):
                corr[i, j] /= np.sqrt(cov[i, i] * cov[j, j])
        return corr

    use_posterior = False

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

    corr_ijt = np.empty((STEPS, N_MODEL, N_MODEL))
    corr_ijt[:, :, :] = np.nan
    cov_ijt = np.empty((STEPS, N_MODEL, N_MODEL))
    cov_ijt[:, :, :] = np.nan
    r = getr()

    for it in range(STEPS // 2, STEPS):
        if it % AINT == 0:
            if use_posterior:
                fcst = hist_fcst[it, :, :].copy()
            else:
                fcst = hist_fcst[it - AINT, :, :].copy()
                for jt in range(AINT):
                    for k in range(nmem):
                        fcst[k, :] = model.timestep(fcst[k, :], DT)
            for i in range(N_MODEL):
                for j in range(N_MODEL):
                    # a38p40
                    vector_i = np.copy(fcst[:, i])
                    vector_j = np.copy(fcst[:, j])
                    vector_i[:] -= np.mean(vector_i)
                    vector_j[:] -= np.mean(vector_j)
                    numera = np.sum(vector_i * vector_j)
                    denomi = (np.sum(vector_i ** 2) * np.sum(vector_j ** 2)) ** 0.5
                    corr_ijt[it, i, j] = numera / denomi
                    cov_ijt[it, i, j] = numera / (nmem - 1.0)
            cov_instant_ij = cov_ijt[it, :, :].copy()

    corr_mean_ij = np.nanmean(corr_ijt, axis=0)
    corr_rms_ij = np.sqrt(np.nanmean(corr_ijt ** 2, axis=0))
    cov_mean_ij = np.nanmean(cov_ijt, axis=0)
    cov_rms_ij = np.sqrt(np.nanmean(cov_ijt ** 2, axis=0))
    ri = np.linalg.inv(getr())
    bhhtri_rms_ij = cov_rms_ij.dot(ri)
    bhhtri_mean_ij = cov_mean_ij.dot(ri)
    rand_ij = np.random.randn(N_MODEL, N_MODEL)
    corr_instant_ij = cov_to_corr(cov_instant_ij)

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
    corr_clim_ij = cov_to_corr(cov_clim_ij)

    data_hash = {"correlation-mean": corr_mean_ij, "correlation-rms": corr_rms_ij, "covariance-mean": cov_mean_ij,
                 "covariance-rms": cov_rms_ij, "covariance-clim": cov_clim_ij, "correlation-clim": corr_clim_ij,
                 "covariance-instant": cov_instant_ij, "correlation-instant": corr_instant_ij}
    # "BHHtRi-mean":bhhtri_mean_ij, "BHHtRi-rms":bhhtri_rms_ij, "random":rand_ij,
    for name in data_hash:
        plot_matrix(data_hash[name], title=name, xlabel="grid index i", ylabel="grid index j", logscale=True,
                    linthresh=1e-1)
        plot_matrix(data_hash[name], title=(name + "_linear"), xlabel="grid index i", ylabel="grid index j",
                    logscale=False)
        print(name)
        matrix_order(np.abs(data_hash[name]), name)

    return 0


def matrix_order(mat_ij_in, name, prioritize_diag=False, max_odr=81):
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
    plt.savefig("histogram_%s.pdf" % name)
    plt.clf()

    return 0


obtain_stats_etkf()
