#!/bin/bash

ls /tmp
cd /tmp/repos/works/couple_lorenz63
git pull
make

commit=`git show HEAD | head -n1 | cut -c8-14`
DATE=`date "+%Y%m%d_%H%M%S"`
title="${DATE}_${commit}_${BATCH_JOB_NAME}"

cp -f data/lyapunov.txt image/true/
cp -f latex/out.pdf image/
aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/${title} --recursive
aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/${title}.pdf
