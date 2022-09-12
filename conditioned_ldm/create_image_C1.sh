#!/bin/bash
#
# A simple script to build the distributed Docker image.
#
# $ create_image_C1.sh
set -ex
TAG=conditioned_ldm

docker build --network=host -f ./Dockerfile.c1 --tag "${TAG}" .
