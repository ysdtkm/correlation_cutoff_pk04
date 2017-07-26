#!/usr/bin/env python3
# This script must be common among jobs.

import os, subprocess, sys

def main():
  commit   = os.getenv("BATCH_COMMIT")
  wdir     = os.getenv("BATCH_WDIR")
  job_name = os.getenv("BATCH_JOB_NAME", "dummy_job")
  commands = sys.argv[1:]

  os.chdir(wdir)
  subprocess.check_call(["git", "pull"])
  subprocess.check_call(["git", "checkout", commit])
  subprocess.check_call(commands)

main()
