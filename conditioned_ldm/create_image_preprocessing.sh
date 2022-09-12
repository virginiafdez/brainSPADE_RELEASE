#!/bin/bash
#
# Script to build the Docker image to train the condldm_models.
#
# $ create_dockerimage_preprocessing.sh
set -ex
TAG=brainspadepreprocess

docker build --network=host --tag "10.202.67.207:5000/${USER}:${TAG}" -f ./Dockerfile.preprocessing . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

docker push "10.202.67.207:5000/${USER}:${TAG}"
