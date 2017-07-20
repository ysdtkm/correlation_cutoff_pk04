#!/usr/bin/env python3

import os, subprocess, sys

os.chdir("/tmp/repos/works/couple_lorenz63")
commit = os.getenv("BATCH_COMMIT", "dummy")
job_name = os.getenv("BATCH_JOB_NAME", "dummy_job")
subprocess.check_call(["git", "pull"])
subprocess.check_call(["git", "checkout", commit])
subprocess.check_call("make")

os.system("cp -f data/lyapunov.txt image/true/")
os.system("cp -f latex/out.pdf image/")
os.system("aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/%s --recursive" % job_name)
os.system("aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/%s.pdf" % job_name)

