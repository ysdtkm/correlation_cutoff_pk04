#!/bin/bash

cd /tmp/repos/works/couple_lorenz63
git pull; git checkout ${BATCH_COMMIT}
make

DATE=`date "+%Y%m%d-%H%M"`
title="${DATE}_${BATCH_COMMIT}_${BATCH_JOB_NAME}"

cp -f data/lyapunov.txt image/true/
cp -f latex/out.pdf image/
aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/${title} --recursive
aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/${title}.pdf

git show HEAD | head -n1
