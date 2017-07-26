#!/usr/bin/env python3

import os, subprocess, sys, re, socket
import numpy as np

# ====================================
from_template = False
if from_template:
  param1s = ["4", "5", "6"]
  param2s = ["covariance-rms", "covariance-mean"]
  param3s = ["9"] # list(map(str, np.linspace(9, 81, 3, dtype=np.int32)))
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
  if not flag_local:
    os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s --recursive" % job_name)
    os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/pdf/%s.pdf" % job_name)

def exec_from_template(param1s, param2s, param3s, job_name, flag_local):
  for param1 in param1s:
    for param2 in param2s:
      write_const_file_from_template(param1, param2, param3s)
      try:
        subprocess.check_call(["make"])
      except:
        subprocess.check_call(["make", "plot"])
      os.system("cp -f data/lyapunov.txt image/true/")
      os.system("cp -f latex/out.pdf image/")
      if not flag_local:
        os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s/%s_%s --recursive"
          % (job_name, param1, param2))
        os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/pdf/%s/%s_%s.pdf"
          % (job_name, param1, param2))
      os.system("rm -rf image_%s_%s" % (param1, param2))
      os.system("mv image image_%s_%s" % (param1, param2))

  sys.path.append('Py')
  import super_verif
  os.system("mkdir -p verif")
  super_verif.verif(param1s, param2s, param3s)
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
