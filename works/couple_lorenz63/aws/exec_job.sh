#!/bin/bash
set -e

git status --short --branch
if [ $# -lt 1 ]; then
  echo -e "\e[33mType (Ctrl-C) to stop.\e[m"
  echo ""
  echo -ne "input \e[33mJOB NAME\e[m: "
  read JOBNAME
else
  echo -e "\e[33mType Enter to continue. Type (Ctrl-C) to stop.\e[m"
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

git pull; git push
echo ""
COMMIT=`git show HEAD | head -n1 | cut -c8-14`

cat <<EOF > ./aws/env.json
{
  "vcpus": 1,
  "memory": 5000,
  "command": ["./myjob.sh"],
  "environment": [
    {
      "name": "BATCH_FILE_TYPE",
      "value": "script"
    },
    {
      "name": "BATCH_FILE_S3_URL",
      "value": "s3://ysdtkm-bucket-1/myjob.sh"
    },
    {
      "name": "BATCH_JOB_NAME",
      "value": "${JOBNAME}"
    },
    {
      "name": "BATCH_COMMIT",
      "value": "${COMMIT}"
    }
  ]
}
EOF

aws s3 cp aws/myjob.sh s3://ysdtkm-bucket-1/
DATE=`date "+%Y%m%d_%H%M"`
id=`aws batch submit-job \
  --job-name ${DATE}_${JOBNAME} \
  --job-queue ${queue} \
  --job-definition def-with-other-image:9 \
  --container-overrides file://aws/env.json | grep jobId`

idcut=`python -c "import sys; print(sys.argv[2])" ${id}`

echo "type following to obtain result"
echo "aws batch describe-jobs --jobs ${idcut} | grep status"
echo ""

rm -f ./aws/env.json
