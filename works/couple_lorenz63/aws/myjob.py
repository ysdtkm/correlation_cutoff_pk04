#!/usr/bin/env python3

import os, subprocess, sys, re

# ====================================
# largest param is innermost paramter

from_template = True
flag_test = False
param1s = ["full", "3-components"]
param2s = ["1.05", "1.1"]
# ====================================

if flag_test:
  os.chdir("/home/tak/repos/works/couple_lorenz63")
else:
  os.chdir("/tmp/repos/works/couple_lorenz63")

commit = os.getenv("BATCH_COMMIT", "dummy")
job_name = os.getenv("BATCH_JOB_NAME", "dummy_job")

subprocess.check_call(["git", "pull"])
# subprocess.check_call(["git", "checkout", commit])

if from_template:
  for param1 in param1s:
    for param2 in param2s:
      rf = open("aws/template_const.py", "r")
      wf = open("Py/const.py", "w")
      for line in rf:
        tmp = re.sub("<<param1>>", param1, line)
        tmp = re.sub("<<param2>>", param2, tmp)
        wf.write(tmp)
      rf.close()
      wf.close()

      subprocess.check_call("make")

      os.system("cp -f data/lyapunov.txt image/true/")
      os.system("cp -f latex/out.pdf image/")
      os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s_%s_%s --recursive" % (job_name, param1, param2))
      os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s_%s_%s.pdf" % (job_name, param1, param2))
      os.system("rm -rf image_%s_%s" % (param1, param2))
      os.system("mv image image_%s_%s" % (param1, param2))

  from Py import super_verif
  os.system("mkdir -p verif")
  super_verif.verif(param1s, param2s)
  os.system("aws s3 cp verif s3://ysdtkm-bucket-1/couple_lorenz63/%s_all --recursive" % job_name)

else:
  subprocess.check_call("make")
  os.system("cp -f data/lyapunov.txt image/true/")
  os.system("cp -f latex/out.pdf image/")
  os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s --recursive" % job_name)
  os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s.pdf" % job_name)

