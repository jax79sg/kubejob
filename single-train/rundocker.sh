#!/bin/bash
docker run -it --gpus all --env-file env.list image-classification-single
