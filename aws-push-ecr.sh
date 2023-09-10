
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 518748727406.dkr.ecr.eu-central-1.amazonaws.com
docker build -t ml/image-de-skew .
docker tag ml/image-de-skew:latest 518748727406.dkr.ecr.eu-central-1.amazonaws.com/ml/image-de-skew:latest
docker push 518748727406.dkr.ecr.eu-central-1.amazonaws.com/ml/image-de-skew:latest
