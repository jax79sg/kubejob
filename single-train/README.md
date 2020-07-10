# Training a ML/DL model on AI Platform (Kubernetes) - Part I
## Why do we do this?
For most of us, we have been building and training our models on our own GPU enabled machines. Our single GPU enabled machines at best can cater up to 11GB of RAM, while this is sufficient for smaller models, its a challenge if we want to train larger models from scratch (E.g. BERT). Beside the scale, the speed of training on Ti1080s and RTX2080s are limited, so moving the training onto Kubernetes where V100 GPUs are available will significantly improve the above.

## What will be achieved at the end of this article?
Following this article, you should get acquainted with the use of Docker and Kubernetes. You would be able to submit jobs to Kubernetes and get the results from S3 object stores.

## Overview: Preparation to Running the training
1. 
## Step-by-Step
## Looking forward
