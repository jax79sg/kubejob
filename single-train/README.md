# Training a ML/DL model on AI Platform (Kubernetes) - Part I
## Why do we do this?
For most of us, we have been building and training our models on our own GPU enabled machines. Our single GPU enabled machines at best can cater up to 11GB of RAM, while this is sufficient for smaller models, its a challenge if we want to train larger models from scratch (E.g. BERT). Beside the scale, the speed of training on Ti1080s and RTX2080s are limited, so moving the training onto Kubernetes where V100 GPUs are available will significantly improve the above.

## What will be achieved at the end of this article?
Following this article, you should get acquainted with the use of Docker and Kubernetes. You would be able to submit jobs to Kubernetes and get the results from S3 object stores.

## Overview: Preparation to Running the training

### Prerequisites
1. A client machine configured to connect to the Kubernetes cluster is available.
2. Docker installed on your own computer

### On your own computer
1. Prepare your training codes
2. Prepare Dockerfile
3. Build a docker image
4. Export/Save the docker image as a file

### On the Kubernetes client
1. Load the docker image file as a docker image
2. Push the docker image to the Docker Registry
3. Run the job on kubernetes

## Step-by-Step
## Looking forward
