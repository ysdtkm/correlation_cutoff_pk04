#!/usr/bin/env python3

import os, subprocess, sys, re, socket
import numpy as np

# ====================================
from_template = True
if from_template:
  param1s = ["3", "4", "5", "6"]
  param2s = ["correlation-rms", "covariance-rms"]
  # param2s = ["correlation-rms", "correlation-mean", "covariance-rms", "covariance-mean",
  #            "BHHtRi-mean", "BHHtRi-rms", "covariance-clim", "correlation-clim"]
  # param3s = ["9", "10", "81"]
  param3s = list(map(str, range(9, 82)))
# ====================================

def main():
  flag_local = (socket.gethostname()[:7] == "DESKTOP")
  job_name = sys.argv[1] if len(sys.argv) > 1 else "dummy_job"
  if not from_template:
    exec_not_from_template(job_name, flag_local)
  else:
    exec_from_template(param1s, param2s, param3s, job_name, flag_local)

def exec_not_from_template(job_name, flag_local):
  subprocess.check_call("make")
  os.system("cp -f data/lyapunov.txt image/true/")
  os.system("cp -f latex/out.pdf image/")
  os.system("tar -czf %s.tar.gz image" % job_name)
  if not flag_local:
    os.system("aws s3 cp %s.tar.gz s3://ysdtkm-bucket-1/couple_lorenz63/tar/" % job_name)
    os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/pdf/%s.pdf" % job_name)

def exec_from_template(param1s, param2s, param3s_raw, job_name, flag_local):
  sys.path.append('Py')
  import super_verif, stats_const

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
      os.system("tar -czf %s_%s.tar.gz image" % (param1, param2))
      os.system("rm -rf image_%s_%s" % (param1, param2))
      os.system("mv image image_%s_%s" % (param1, param2))
      if not flag_local:
        os.system("aws s3 cp %s_%s.tar.gz s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s/"
          % (param1, param2, job_name))
        os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/pdf/%s/%s_%s.pdf"
          % (job_name, param1, param2))

  os.system("mkdir -p verif")
  super_verif.verif(param1s, param2s, param3s_raw, param3s_arr)
  if not flag_local:
    os.system("aws s3 sync verif s3://ysdtkm-bucket-1/couple_lorenz63/pdf/%s" % job_name)

def write_const_file_from_template(param1, param2, param3s):
  rf = open("aws/template_const.py", "r")
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

def sanitize_num(strin):
  tmp = strin
  tmp = re.sub("\"", "", tmp)
  tmp = re.sub("\.", "", tmp)
  return tmp

main()
