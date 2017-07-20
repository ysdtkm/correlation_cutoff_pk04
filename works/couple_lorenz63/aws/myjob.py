#!/usr/bin/env python3

import os, subprocess, sys, re

os.chdir("/tmp/repos/works/couple_lorenz63")
commit = os.getenv("BATCH_COMMIT", "dummy")
job_name = os.getenv("BATCH_JOB_NAME", "dummy_job")
from_template = os.getenv("BATCH_FROM_TEMPLATE")

subprocess.check_call(["git", "pull"])
subprocess.check_call(["git", "checkout", commit])

if from_template == "True":
  rhos = ["1.05", "1.1"]
  for rho in rhos:
    rf = open("aws/template_const.py", "r")
    wf = open("Py/const.py", "w")
    for line in rf:
      wf.write(re.sub("<<param1>>", rho, line))
    rf.close()
    wf.close()

    subprocess.check_call("make")
    os.system("cp -f data/lyapunov.txt image/true/")
    os.system("cp -f latex/out.pdf image/")
    os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s_%s --recursive" % (job_name, rho))
    os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s_%s.pdf" % (job_name, rho))
    os.system("mv -f image imege_%s" % rho)

  from Py import super_verif
  os.system("mkdir -p image")
  super_verif.verif(rhos)
  os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/%s_all --recursive" % job_name)

else:
  subprocess.check_call("make")
  os.system("cp -f data/lyapunov.txt image/true/")
  os.system("cp -f latex/out.pdf image/")
  os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s --recursive" % job_name)
  os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s.pdf" % job_name)

