#!/bin/bash

ls /tmp
cd /tmp/repos/works/couple_lorenz63
git pull
make plot

commit=`git show HEAD | head -n1 | cut -c8-14`
DATE=`date "+%Y%m%d_%H%M%S"`
title="${DATE}_${commit}_${BATCH_JOB_NAME}"

aws s3 cp image s3://ysdtkm-bucket-1/couple_lorenz63/tar/${title} --recursive
# tar -czvf ${title}.tar.gz image

make tex
aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/tar/${title}/
aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/${title}.pdf
