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

if [ $# -ge 2 ]; then
  queue="queue_fast"
else
  queue="queue_slow"
fi

set +e
git add .
git commit -m "${JOBNAME}"
set -e

git pull && git push
echo ""
COMMIT=`git show HEAD | head -n1 | cut -c8-14`
DATE=`date "+%Y%m%d_%H%M"`
SCRIPT="myjob.py"

cat <<EOF > ./aws/env.json
{
  "vcpus": 1,
  "memory": 7000,
  "command": ["./${SCRIPT}", "python", "exec_from_template.py", "${DATE}_${COMMIT}_${JOBNAME}"],
  "environment": [
    {
      "name": "BATCH_FILE_TYPE",
      "value": "script"
    },
    {
      "name": "BATCH_WDIR",
      "value": "/tmp/repos/works/couple_lorenz63"
    },
    {
      "name": "BATCH_FILE_S3_URL",
      "value": "s3://ysdtkm-bucket-1/${SCRIPT}"
    },
    {
      "name": "BATCH_JOB_NAME",
      "value": "${DATE}_${COMMIT}_${JOBNAME}"
    },
    {
      "name": "BATCH_COMMIT",
      "value": "${COMMIT}"
    }
  ]
}
EOF

aws s3 cp aws/${SCRIPT} s3://ysdtkm-bucket-1/
id=`aws batch submit-job \
  --job-name ${DATE}_${COMMIT}_${JOBNAME} \
  --job-queue ${queue} \
  --job-definition def-with-other-image:9 \
  --container-overrides file://aws/env.json | grep jobId`

idcut=`python -c "import sys; print(sys.argv[2])" ${id}`

echo "type following to obtain result"
echo "aws batch describe-jobs --jobs ${idcut} | grep status"
echo ""

rm -f ./aws/env.json
