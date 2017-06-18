#!/bin/bash

ls /tmp
cd /tmp/repos/works/couple_lorenz63
git pull
make

cp -f latex/out.pdf image/
commit=`git show HEAD | head -n1 | cut -c8-14`
DATE=`date "+%Y%m%d_%H%M%S"`
title="${DATE}_${commit}_${BATCH_JOB_NAME}"
tar -czvf ${title}.tar.gz image
aws s3 cp ${title}.tar.gz s3://ysdtkm-bucket-1/couple_lorenz63/tar/
aws s3 cp latex/out.pdf s3://ysdtkm-bucket-1/couple_lorenz63/${title}.pdf
