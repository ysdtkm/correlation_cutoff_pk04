#!/usr/bin/env python3

import os, subprocess

def main():
  os.chdir("/tmp/repos/works/couple_lorenz63")
  commit = os.getenv("BATCH_COMMIT", "dummy")
  job_name = os.getenv("BATCH_JOB_NAME", "dummy_job")
  subprocess.check_call(["git", "pull"])
  subprocess.check_call(["git", "checkout", commit])
  subprocess.check_call(["python", "aws/exec_from_template.py", job_name])

main()
