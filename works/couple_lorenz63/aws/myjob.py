#!/usr/bin/env python3

import os, subprocess, sys, re, socket
import numpy as np

# ====================================
from_template = True

if from_template: # largest param is innermost paramter
  param1s = ["4", "5", "6"]
  param2s = ["covariance-rms", "covariance-mean"]
  # param2s = ["covariance-rms", "covariance-mean", "correlation-rms", "correlation-mean", "bhhtri-rms", "bhhtri-mean"]
  param3s = list(map(str, np.linspace(9, 81, 3, dtype=np.int32)))
  # param3s = list(map(str, np.linspace(9, 81, 19, dtype=np.int32)))
# ====================================

flag_test = (socket.gethostname()[:7] == "DESKTOP")
if flag_test:
  param1s = param1s[:1]
  param2s = param2s[:1]
  # param3s = param3s[:1]

def main():
  if flag_test:
    os.chdir("/home/tak/repos/works/couple_lorenz63")
  else:
    os.chdir("/tmp/repos/works/couple_lorenz63")

  commit = os.getenv("BATCH_COMMIT", "dummy")
  job_name = os.getenv("BATCH_JOB_NAME", "dummy_job")

  subprocess.check_call(["git", "pull"])
  if not flag_test:
    subprocess.check_call(["git", "checkout", commit])

  if from_template:
    exec_from_template(param1s, param2s, param3s, job_name, flag_test)
  else:
    exec_not_from_template(job_name, flag_test)

def write_const_file_from_template(param1s, param2s, param3s):
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

def exec_from_template(param1s, param2s, param3s, job_name, flag_test):
  for param1 in param1s:
    for param2 in param2s:
      write_const_file_from_template(param1s, param2s, param3s)
      try:
        subprocess.check_call(["make"])
      except:
        subprocess.check_call(["make", "plot"])
      os.system("cp -f data/lyapunov.txt image/true/")
      os.system("cp -f latex/out.pdf image/")
      if not flag_test:
        os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s/%s_%s --recursive"
          % (job_name, param1, param2))
        os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s/%s_%s.pdf"
          % (job_name, param1, param2))
      os.system("rm -rf image_%s_%s" % (param1, param2))
      os.system("mv image image_%s_%s" % (param1, param2))

  sys.path.append('Py')
  import super_verif
  os.system("mkdir -p verif")
  super_verif.verif(param1s, param2s, param3s)
  if not flag_test:
    os.system("aws s3 sync verif s3://ysdtkm-bucket-1/couple_lorenz63/%s" % job_name)

def exec_not_from_template(job_name, flag_test):
  subprocess.check_call("make")
  os.system("cp -f data/lyapunov.txt image/true/")
  os.system("cp -f latex/out.pdf image/")
  if not flag_test:
    os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s --recursive" % job_name)
    os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s.pdf" % job_name)

def sanitize_num(strin):
  tmp = strin
  tmp = re.sub("\"", "", tmp)
  tmp = re.sub("\.", "", tmp)
  return tmp

main()
