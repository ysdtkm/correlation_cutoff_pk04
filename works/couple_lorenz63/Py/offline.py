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

    def plot_time_corr(data_rms, data_mean, data_clim, name, i, j, delta_t_set):
        plt.plot(delta_t_set, data_rms[:, i, j], label="RMS")
        plt.plot(delta_t_set, data_mean[:, i, j], label="Mean")
        plt.plot(delta_t_set, data_clim[:, i, j], label="Clim")
        img_dir = "offline/time"
        os.makedirs(img_dir, exist_ok=True)
        vars = ["x_e", "y_e", "z_e", "x_t", "y_t", "z_t", "X", "Y", "Z"]
        plt.title("lagged correlation: %s vs %s" % (vars[i], vars[j]))
        plt.xlabel("lagged time (steps): positive means %s leads %s" % (vars[i], vars[j]))
        plt.ylabel("correlation")
        plt.ylim(-1.2, 1.2)
        plt.legend()
        plt.axhline(y=0.0, color="black", alpha=0.5)
        plt.savefig("./%s/time_%s.pdf" % (img_dir, name))
        plt.close()

    def plot_matrix(data, img_dir, name="", title="", color=plt.cm.bwr, xlabel="", ylabel="",
                    logscale=False, linthresh=1e-3, cmax=None):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
        if cmax is None:
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
        plt.colorbar(map1)
        plt.title(title)
        plt.gca().invert_yaxis()
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig("./%s/matrix_%s_%s.pdf" % (img_dir, name, title))
        plt.close()
        return 0

    def plot_covs_corrs(mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij, corr_clim_ij, cov_clim_ij, num_delta_t, delta_t_set):
        data_hash = {"correlation-mean": mean_corr_ij, "correlation-rms": rms_corr_ij,
                     "covariance-mean": mean_cov_ij, "covariance-rms": rms_cov_ij,
                     "covariance-clim": cov_clim_ij, "correlation-clim": corr_clim_ij}

        for delta_t in range(num_delta_t):
            for name in data_hash:
                data = data_hash[name][delta_t, :, :]
                name2 = name + "_" + str(delta_t)
                img_dir = "offline/del_t_%d" % delta_t
                cmax = 1.0 if "corr" in name else None
                plot_matrix(data, img_dir, title=name2, xlabel="grid index i",
                            ylabel="grid index j", logscale=True, linthresh=1e-1, cmax=cmax)
                plot_matrix(data, img_dir, title=(name2 + "_linear"), xlabel="grid index i",
                            ylabel="grid index j", logscale=False, cmax=cmax)
                print(name2)
                # matrix_order(np.abs(data), img_dir, name2)

        set_ij = [(1, 1), (4, 4), (7, 7), (1, 4), (4, 7), (1, 7)]
        for i, j in set_ij:
            data_rms  = np.concatenate((np.transpose( rms_corr_ij[:0:-1,:,:], axes=(0, 2, 1)),  rms_corr_ij[:,:,:]), axis=0)
            data_mean = np.concatenate((np.transpose(mean_corr_ij[:0:-1,:,:], axes=(0, 2, 1)), mean_corr_ij[:,:,:]), axis=0)
            data_clim = np.concatenate((np.transpose(corr_clim_ij[:0:-1,:,:], axes=(0, 2, 1)), corr_clim_ij[:,:,:]), axis=0)
            x_list = list(map(lambda x: -x, delta_t_set[:0:-1])) + delta_t_set[:]
            name = "corr_%d_%d" % (i, j)
            plot_time_corr(data_rms, data_mean, data_clim, name, i, j, x_list)

    hist_fcst, nature, nmem = obtain_cycle()

    num_delta_t = 6
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

    plot_covs_corrs(mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij, clim_corr_ij, clim_cov_ij, num_delta_t, delta_t_set)


def matrix_order(mat_ij_in, img_dir, name, prioritize_diag=False, max_odr=81):
    if not (len(mat_ij_in.shape) == 2 and mat_ij_in.shape[0] == mat_ij_in.shape[1]):
        raise Exception("input matrix non-square")
    else:
        n = mat_ij_in.shape[0]
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
    os.system("mkdir -p %s" % img_dir)
    plt.savefig("%s/histogram_%s.pdf" % (img_dir, name))
    plt.clf()

    return 0


if __name__ == "__main__":
    obtain_stats_etkf()
