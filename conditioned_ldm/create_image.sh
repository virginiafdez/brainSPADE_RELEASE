#!/bin/bash
#
# A simple script to build the distributed Docker image.
#
# $ create_docker_image.sh
set -ex
TAG=condldm

docker build --network=host --tag "10.202.67.207:5000/${USER}:${TAG}" . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

docker push "10.202.67.207:5000/${USER}:${TAG}"
