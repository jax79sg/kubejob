#!/bin/bash
docker build . -t image-classification-multi
docker tag image-classification-multi myregistry.com:5000/image-classification-multi
docker push myregistry.com:5000/image-classification-multi 
