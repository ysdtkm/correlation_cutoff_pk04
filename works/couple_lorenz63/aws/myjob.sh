#!/bin/bash

# date
# echo "Args: $@"
# env
# echo "This is my simple test job!."
# echo "jobId: $AWS_BATCH_JOB_ID"
# sleep $1
# date
# echo "bye bye!!"

ls /tmp
cd /tmp/repos/works/couple_lorenz63
git pull
make

cp -f latex/out.pdf image/
commit=`git show HEAD | head -n1 | cut -c8-14`
DATE=`date "+%Y%m%d_%H%M%S"`
tar -czvf ${DATE}_${commit}.tar.gz image
aws s3 cp ${DATE}_${commit}.tar.gz s3://ysdtkm-bucket-1/couple_lorenz63/
