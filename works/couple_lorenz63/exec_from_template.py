#!/usr/bin/env python3

import os
import subprocess
import sys
import re
import socket

# ====================================
from_template = True
if from_template:
    param1s = ["4", "6"]
    param2s = ["correlation-rms", "covariance-rms"]
    # param2s = ["correlation-rms", "correlation-mean", "covariance-rms", "covariance-mean",
    #            "BHHtRi-mean", "BHHtRi-rms", "covariance-clim", "correlation-clim"]
    param3s = ["9"] # , "10", "81"]  # list(map(str, range(9, 82)))
# ====================================

RAW_DIR = "raw"
TAR_DIR = "tar"

def main():
    flag_local = (socket.gethostname()[:7] == "DESKTOP")
    job_name = sys.argv[1] if len(sys.argv) > 1 else "dummy_job"
    os.system("rm -rf %s %s" % (RAW_DIR, TAR_DIR))
    os.system("mkdir -p %s %s" % (RAW_DIR, TAR_DIR))

    if not from_template:
        exec_not_from_template(job_name, flag_local)
    else:
        exec_from_template(param1s, param2s, param3s, job_name, flag_local)


def exec_not_from_template(job_name, flag_local):
    subprocess.check_call(["make", "tex"])
    os.system("cp -f data/lyapunov.txt image/true/")
    os.system("cp -f latex/out.pdf image/")
    os.system("mv -f image %s/" % TAR_DIR)
    os.system("cp -f latex/out.pdf %s/%s.pdf" % (RAW_DIR, job_name))


def exec_from_template(param1s, param2s, param3s_raw, job_name, flag_local):
    def sanitize_num(strin):
        tmp = strin
        tmp = re.sub("\"", "", tmp)
        tmp = re.sub("\.", "", tmp)
        return tmp

    def write_const_file_from_template(param1, param2, param3s):
        rf = open("Py/template_const.py", "r")
        wf = open("Py/const.py", "w")
        for line in rf:
            tmp = re.sub("<<param1>>", param1, line)
            tmp = re.sub("<<param2>>", param2, tmp)
            if "<<param3>>" in tmp:
                for param3 in param3s:
                    tmp2 = re.sub("<<param3>>", param3, tmp)
                    wf.write(re.sub("<<param3_sanit>>", sanitize_num(param3), tmp2))
            else:
                wf.write(tmp)
        rf.close()
        wf.close()

    sys.path.append('Py')
    import super_verif
    import stats_const

    param3s_arr = [[[] for param2 in param2s] for param1 in param1s]
    for i, param1 in enumerate(param1s):
        for j, param2 in enumerate(param2s):
            weight_order = stats_const.stats_order(param2).flatten()
            param3s = [p3 for p3 in param3s_raw if (int(p3) - 1 in weight_order)]
            param3s_arr[i][j] = param3s

            write_const_file_from_template(param1, param2, param3s)
            subprocess.check_call(["make", "plot"])
            try:
                subprocess.check_call(["make", "tex"])
            except CalledProcessError:
                pass
            os.system("cp -f data/lyapunov.txt image/true/")
            os.system("cp -f latex/out.pdf image/")
            os.system("mv -f image %s/image_%s_%s" % (TAR_DIR, param1, param2))
            os.system("cp -f latex/out.pdf %s/%s_%s.pdf" % (RAW_DIR, param1, param2))

    os.system("mkdir -p verif")
    super_verif.verif(param1s, param2s, param3s_raw, param3s_arr)
    os.system("mv -f verif %s/" % RAW_DIR)


main()
