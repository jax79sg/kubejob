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
### On your own computer
#### Prepare your training codes
A sample of the training code is found in [image_classification_single.py](https://raw.githubusercontent.com/jax79sg/kubejob/master/single-train/image_classification_single.py). 

To pull the data and to save the results, you would need to do it on a shared external storage. A temporary MINIO server has been setup in the AI Platform, your training codes should pull and save the data there. The following extracts the related codes.

This setups the helper codes and pulls the relevant parameters about the S3. 
The parameters are to be sent into the environment variables. You may also hard code the variables if that suits you, but i would encourage you to use either environment variables or argparse.
```python
### Setup of S3 parameters and helper functions
trainingbucket= os.environ['trainingbucket'] #'training'
datasetsbucket= os.environ['datasetsbucket'] #'datasets'
import boto3
from botocore.client import Config
s3 = boto3.resource('s3',
                    endpoint_url= os.environ['endpoint_url'] ,
                    aws_access_key_id= os.environ['aws_access_key_id'] ,
                    aws_secret_access_key= os.environ['aws_secret_access_key'],
                    config=Config(signature_version= os.environ['signature_version']),
                    region_name= os.environ['region_name'])

def s3_download_file(localfile,bucket,s3path):
    print("S3 Download s3://"+bucket+"/" + s3path + " to " + localfile )
    s3.Bucket(bucket).download_file(s3path,localfile)
    
def s3_upload_file(localfile,bucket,s3path):
    print("S3 Uploading " + localfile + " to s3://"+bucket + s3path+localfile)
    s3.Bucket(bucket).upload_file(localfile,s3path+localfile)
    
def s3_upload_folder(folder, bucket,s3path):
    
    from glob import glob
    print("Processing folder")
    for file in glob(folder+"/**/*",recursive=True):
      if (os.path.isdir(file)) == False:  
        print("Processing " + file)
        s3_upload_file(bucket='training',localfile=file,s3path='')
```
This part of the code will save the results into S3.
```python
### Up until this point,allthe model files are saved on container. After this container finishes execution, the files will be gone.
### Start saving the checkpoints and model files.
import json
with open('catdogclassification.json', 'w') as fp:
    json.dump(history.history, fp)
s3_upload_file(bucket=trainingbucket,localfile='catdogclassification.json',s3path='')

s3_upload_folder(bucket=trainingbucket,folder='catdogclassification_model',s3path='')

for epochrun in range(epochs):
    s3_upload_file(bucket=trainingbucket,localfile='catdogclassification_save_at_'+str(epochrun+1)+'.h5',s3path='')

```

## Looking forward
