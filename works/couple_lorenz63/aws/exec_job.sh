#!/bin/bash

git s
echo -n "input something if you want to avoid auto-commit: "
read INPUT
if [ "${INPUT}" = "" ]; then
  git cam mod; git pull; git push
else
  exit 1
fi

aws s3 cp aws/myjob.sh s3://ysdtkm-bucket-1/
DATE=`date "+%Y%m%d-%H%M%S"`
id=`aws batch submit-job \
  --job-name batch_python_${DATE} \
  --job-queue queue_slow \
  --job-definition def-with-other-image:9 \
  --container-overrides file://aws/env.json | grep jobId`

idcut=`python -c "import sys; print(sys.argv[2])" ${id}`

echo "type following to obtain result"
echo "aws batch describe-jobs --jobs ${idcut} | grep status"
