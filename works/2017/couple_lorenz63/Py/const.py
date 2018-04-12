#!/usr/bin/env python

import numpy as np

RAW_DIR = "raw"
TAR_DIR = "tar"

# note: "dimc" in comments is dimension of component (for non/weakly coupled)
N_MODEL = 9  # dimension of model variable n
P_OBS = N_MODEL  # dimension of observation variable m
N_ATM = 6
P_ATM = N_ATM

DT = 0.01
TMAX = 3
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR_A = 1.0
OERR_O = 5.0
FERR_INI = 10.0
AINT = 8

rho = "adaptive"
nmem = 4
amp_b_tdvar = 2.0
amp_b_fdvar = 1.5

EXPLIST = [
    dict(name="Full", rho=rho, nmem=nmem, method="etkf", couple="strong", r_local="full"),
    dict(name="Adjacent", rho=rho, nmem=nmem, method="etkf", couple="strong", r_local="adjacent"),
    dict(name="ENSO-coupling", rho=rho, nmem=nmem, method="etkf", couple="strong", r_local="enso_coupling"),
    dict(name="Atmos-coupling", rho=rho, nmem=nmem, method="etkf", couple="strong", r_local="atmos_coupling"),
    dict(name="Individual", rho=rho, nmem=nmem, method="etkf", couple="strong", r_local="individual"),
]

# dict(name="etkf", rho=rho, nmem=nmem, method="etkf", couple="strong", r_local="full"),
# dict(name="tdvar", nmem=1, method="3dvar", couple="strong", amp_b=amp_b_tdvar),
# dict(name="fdvar", nmem=1, method="4dvar", couple="strong", amp_b=amp_b_fdvar),

Calc_lv = False

ETKF_vo = 1.0
ETKF_kappa = 1.01
ETKF_AI_max = 1.2
ETKF_AI_min = 0.9


def getr() -> np.ndarray:
    """
    note: Non-diagonal element in R is ignored in main.exec_obs()
    :return r: [P_OBS, P_OBS]
    """
    r = np.identity(P_OBS) * OERR_A ** 2
    if N_MODEL == 9:
        if P_OBS == N_MODEL:
            for i in range(6, 9):
                r[i, i] = OERR_O ** 2
        else:
            import warnings
            warnings.warn(
                "getr() ignores OERR_O if P_OBS != N_MODEL. P_OBS=%d, N_MODEL=%d was passed." % (P_OBS, N_MODEL))
    return r


def geth() -> np.ndarray:
    """
    :return h: [P_OBS, N_MODEL]
    """
    # h = np.zeros((P_OBS, N_MODEL))
    # for i in range(0, min(N_MODEL, P_OBS)):
    #     h[i, i] = 1.0
    # if P_OBS != N_MODEL:
    #     import warnings
    #     warnings.warn("geth() cannot correctly deal with P_OBS != N_MODEL. P_OBS=%d, N_MODEL=%d was passed."
    #                   % (P_OBS, N_MODEL))
    # h = np.diag([0, 1, 0, 0, 1, 0, 0, 1, 0])  # y-only
    # h = np.diag([0, 0, 1, 0, 0, 1, 0, 0, 1])  # z-only
    h = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1])
    return h


def debug_obj_print(obj, scope):
    print([k for k, v in scope.items() if id(obj) == id(v)])
    print(type(obj))
    print(obj)
    print()
    return 0
