#!/bin/bash

git s
echo ""
echo -n "input something if you want to avoid auto-commit: "
read STOPFLAG

echo ""
echo -ne "input \e[33mJOB NAME\e[m: "
read JOBNAME
echo ""

if [ "${STOPFLAG}" = "" ]; then
  git cam "${JOBNAME}"; git pull; git push
else
  exit 1
fi

echo -n "choose queue type (default: slow): "
read QUEUETYPE
if [ "${QUEUETYPE}" != "" ]; then
  queue="queue_fast"
else
  queue="queue_slow"
fi

cat <<EOF > ./aws/env.json
{
  "vcpus": 2,
  "memory": 2000,
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
    }
  ]
}
EOF

aws s3 cp aws/myjob.sh s3://ysdtkm-bucket-1/
DATE=`date "+%Y%m%d-%H%M%S"`
id=`aws batch submit-job \
  --job-name python_${DATE}_${JOBNAME} \
  --job-queue ${queue} \
  --job-definition def-with-other-image:9 \
  --container-overrides file://aws/env.json | grep jobId`

idcut=`python -c "import sys; print(sys.argv[2])" ${id}`

echo "type following to obtain result"
echo "aws batch describe-jobs --jobs ${idcut} | grep status"
echo ""

rm -f ./aws/env.json
