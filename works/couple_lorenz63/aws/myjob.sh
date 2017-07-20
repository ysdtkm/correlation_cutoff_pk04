#!/bin/bash

cd /tmp/repos/works/couple_lorenz63
git pull; git checkout ${BATCH_COMMIT}
make

cp -f data/lyapunov.txt image/true/
cp -f latex/out.pdf image/
aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/${BATCH_JOB_NAME} --recursive
aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/${BATCH_JOB_NAME}.pdf

git show HEAD | head -n1
