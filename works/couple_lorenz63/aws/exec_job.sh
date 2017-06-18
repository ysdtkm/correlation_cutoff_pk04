aws s3 cp myjob.sh s3://ysdtkm-bucket-1/
DATE=`date "+%Y%m%d-%H%M%S"`
id=`aws batch submit-job \
  --job-name batch_python_${DATE} \
  --job-queue queue_fast \
  --job-definition def-with-other-image:9 \
  --container-overrides file://env.json | grep jobId`

idcut=`python -c "import sys; print(sys.argv[2])" ${id}`

echo "type following to obtain result"
echo "aws batch describe-jobs --jobs ${idcut} | grep status"
