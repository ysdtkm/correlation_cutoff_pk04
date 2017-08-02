#!/bin/bash
set -e

git status --short --branch .
if [ $# -lt 1 ]; then
  echo "Type (Ctrl-C) to stop."
  echo ""
  echo -n "input JOB NAME: "
  read JOBNAME
else
  echo "Type Enter to continue. Type (Ctrl-C) to stop."
  read DUMMY
  JOBNAME=$1
fi

set +e
git add .
git commit -m "${JOBNAME}"
set -e

git pull && git push
echo ""
COMMIT=`git show HEAD | head -n1 | cut -c8-14`
DATE=`date "+%Y%m%d_%H%M"`
JOBNAME2=${DATE}_${COMMIT}_${JOBNAME}
WDIR=/homes/metogra/tyoshida/shrt/submit
WDIR_IN_REPOS=works/couple_lorenz63
COMMANDS="./exec_from_template.py"
# COMMANDS="make clean all"

ssh tyoshida@halo.atmos.umd.edu -t "nohup bash /homes/metogra/tyoshida/repos/works/submit_halo/exec_halo.sh ${COMMIT} ${JOBNAME2} \$\$ ${WDIR} ${WDIR_IN_REPOS} ${COMMANDS} >& ~/shrt/log/${JOBNAME2}_\$\$.log &"
