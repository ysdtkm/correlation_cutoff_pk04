#!/usr/bin/env python

import os
import numpy as np
from const import RAW_DIR, TAR_DIR
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_lagged_correlation():
    def plot_time_corr(data_rms, data_mean, data_clim, name, i, j, delta_t_list):
        for target in ["error", "anomaly"]:
            if target == "error":
                plt.plot(delta_t_list, data_rms[:, i, j], label="RMS")
                plt.plot(delta_t_list, data_mean[:, i, j], label="Mean")
            else:
                plt.plot(delta_t_list, data_clim[:, i, j], label="Anomaly")
            img_dir = "%s/time" % RAW_DIR
            os.makedirs(img_dir, exist_ok=True)
            var_names = ["x_e", "y_e", "z_e", "x_t", "y_t", "z_t", "X", "Y", "Z"]
            plt.title("lagged correlation: %s vs %s" % (var_names[i], var_names[j]))
            plt.xlabel("lagged time (steps): positive means %s leads %s" % (var_names[i], var_names[j]))
            plt.ylabel("correlation")
            plt.ylim(-1.2, 1.2)
            plt.legend()
            plt.axhline(y=0.0, color="black", alpha=0.5)
            plt.savefig("./%s/time_%s_%s.pdf" % (img_dir, name, target))
            plt.close()

    def plot_matrix(data, img_dir, name="", title="", color=plt.cm.bwr, xlabel="", ylabel="",
                    logscale=False, linthresh=1e-3, cmax=None):
        plt.rcParams["font.size"] = 16
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0.10, right=0.93, bottom=0.14, top=0.94)
        if cmax is None:
            cmax = np.max(np.abs(data))
        if logscale:
            map1 = ax.pcolor(data, cmap=color, norm=colors.SymLogNorm(linthresh=linthresh * cmax))
        else:
            map1 = ax.pcolor(data, cmap=color)
        if color == plt.cm.bwr:
            map1.set_clim(-1.0 * cmax, cmax)
        elif color == plt.cm.gray_r:
            map1.set_clim(0, cmax)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        ax.set_xlabel(xlabel)
        ax.set_xticks(np.arange(0, data.shape[1]) + 0.5)
        ax.xaxis.set_tick_params(size=0)
        xtlabel = np.arange(1, 10)
        ax.set_xticklabels(xtlabel, rotation=0, fontsize=14)

        ax.set_ylabel(ylabel)
        plt.colorbar(map1)
        # plt.title(title)
        plt.gca().invert_yaxis()
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig("./%s/matrix_%s_%s.pdf" % (img_dir, name, title))
        plt.close()
        return 0

    def plot_covs_corrs(mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij,
                        corr_clim_ij, cov_clim_ij, num_delta_t, delta_t_list, skip_matrix=True):
        data_hash = {"correlation-mean": mean_corr_ij, "correlation-rms": rms_corr_ij,
                     "covariance-mean": mean_cov_ij, "covariance-rms": rms_cov_ij,
                     "covariance-clim": cov_clim_ij, "correlation-clim": corr_clim_ij}

        if not skip_matrix:
            for delta_t in range(num_delta_t):
                for name in data_hash:
                    data = data_hash[name][delta_t, :, :]
                    name2 = name + "_" + str(delta_t)
                    img_dir = "%s/del_t_%d" % (TAR_DIR, delta_t)
                    cmax = 1.0 if "corr" in name else None
                    plot_matrix(data, img_dir, title=name2, xlabel="grid index i",
                                ylabel="grid index j", logscale=True, linthresh=1e-1, cmax=cmax)
                    plot_matrix(data, img_dir, title=(name2 + "_linear"), xlabel="grid index i",
                                ylabel="grid index j", logscale=False, cmax=cmax, color=plt.cm.gray_r)
                    print(name2)
                    matrix_order(np.abs(data), img_dir, name2)

        set_ij = [(1, 1), (4, 4), (7, 7), (1, 4), (4, 7), (1, 7)]
        for i, j in set_ij:
            data_rms = np.concatenate((np.transpose(rms_corr_ij[:0:-1, :, :], axes=(0, 2, 1)),
                                       rms_corr_ij[:, :, :]), axis=0)
            data_mean = np.concatenate((np.transpose(mean_corr_ij[:0:-1, :, :], axes=(0, 2, 1)),
                                        mean_corr_ij[:, :, :]), axis=0)
            data_clim = np.concatenate((np.transpose(corr_clim_ij[:0:-1, :, :], axes=(0, 2, 1)),
                                        corr_clim_ij[:, :, :]), axis=0)
            x_list = list(map(lambda x: -x, delta_t_list[:0:-1])) + delta_t_list[:]
            name = "corr_%d_%d" % (i, j)
            plot_time_corr(data_rms, data_mean, data_clim, name, i, j, x_list)

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

        plt.rcParams["font.size"] = 16
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        x = np.array(range(1, len(sorted_vals) + 1))
        y = np.array(sorted_vals)
        plt.bar(x, y / np.max(y), label=name, color="gray")
        plt.xlabel("descending order")
        plt.ylabel("RMS of correlation")
        # plt.yscale("log")
        plt.xlim(0, 82)
        plt.ylim(0.0, 1.0)
        # plt.legend()
        os.system("mkdir -p %s" % img_dir)
        plt.savefig("%s/histogram_%s.pdf" % (img_dir, name))
        plt.clf()
        plt.close("all")

        return 0

    def load_data():
        datadir = "data/offline"
        names = ["mean_corr", "rms_corr", "mean_cov", "rms_cov",
                 "clim_corr", "clim_cov", "num_delta_t", "delta_t_list"]
        datas = []
        for name in names:
            data = np.load("%s/%s.npy" % (datadir, name))
            datas.append(data)
        return datas

    mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij, \
    clim_corr_ij, clim_cov_ij, num_delta_t, delta_t_list = load_data()
    plot_covs_corrs(mean_corr_ij, rms_corr_ij, mean_cov_ij, rms_cov_ij,
                    clim_corr_ij, clim_cov_ij, int(num_delta_t), list(delta_t_list), skip_matrix=False)


if __name__ == "__main__":
    plot_lagged_correlation()
