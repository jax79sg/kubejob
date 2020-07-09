#!/bin/bash
docker build . -t image-classification-single
docker tag image-classification-single myregistry.com:5000/image-classification-single
docker push  myregistry.com:5000/image-classification-single
