# Kubernetes Series - Part I.
# Training a Open Source ML/DL model on AI Platform.
For a lot of us, we have been building and/or training our models on our own GPU enabled machines. Our single GPU enabled machines at best can cater up to 11GB of RAM, while this is sufficient for smaller models, its a challenge if we want to train larger models from scratch (E.g. BERT). Other than the scale, the speed of training on Ti1080s and RTX2080s are limited, so moving the training onto Kubernetes where V100 GPUs are available will significantly improve the above.

## Who should try this?
### 3rd Parties
If you have a Deep Learning architecture you got from someone or pulled from open sourced research, and you need to perform some form of training on the model without intimate knowledge of the codes, this method would be most suitable for you. 
### Development in Docker
For those who are regularly developing their codes in Docker, this would be very apt for them as well. The advantage of developing DL models in Docker is that they highly flexible when it comes to using different frameworks and versions. For example, you don't have to crack your head on different versions of CUDA on your machine, just make sure you have a docker for every version. Most times, you don't even have to worry about this as the frameworks such as Tensorflow come with their own docker images anyway.

## What will be achieved at the end of this article?
Following this article, you will get acquainted with very basic use of Docker and Kubernetes. You would be able to submit jobs to Kubernetes and get the results from S3 object stores.

## Overview: Preparation and then the model training

### Prerequisites
1. A client machine configured to connect to the Kubernetes cluster.
2. Docker installed on your own computer (Both Windows and Linux versions are fine)

### On your own computer
0. Prepare your datasets
1. Prepare your training codes
2. Prepare Dockerfile file
3. Build a docker image
4. Export/Save the docker image as a file
5. Prepare kubernetes job yaml file
6. Transfer to Kubernetes client

### On the Kubernetes client
0. Load datasets onto S3
1. Load the docker image file as a docker image
2. Push the docker image to the Docker Registry
3. Run the job yaml file

## Step-by-Step
### On your own computer
#### Prepare your datasets
As we are submiting the jobs to the network for training, it means that your datasets needs to be accessible by several computers on the network via a shared storage. On the AI Platform, the following would be made available, S3, NFS and Hadoop. In this article, we will demonstrate with the use of S3 storage as it can cater to both structured and unstructured data.

We would be loading datasets into S3 via a network connection, so its generally more efficient if you transfer files as a zipped archive rather than thousands of individual files unless your training codes do it differently. Do zip up your datasets if you can, however, if you have extremely large datasets, or structured data in databases, you can prepare the data in their native forms.

#### Prepare your training codes
Your training codes should consist mainly of 3 parts.
1. Downloading of datasets from S3
2. Preprocessing and training of model
3. Uploading of models and results data to S3

Finally make sure your codes can run and train for at least an epoch to verify its working.

A sample of a training code is found in [image_classification_single.py](https://raw.githubusercontent.com/jax79sg/kubejob/master/single-train/image_classification_single.py). The only change to this code is such that the zipped datasets would be downloaded from S3 and then extracted for processing, and then the model and results saved in S3. Your situation could be different, please exercise your own considerations. (See Wei Deng's article in part X for load-in-memory example)

A temporary MINIO S3 server has been setup in the AI Platform, your training codes should pull and save the data there. The following extracts the related codes from the above example. This setups the helper codes and pulls the relevant parameters about the S3. 
The parameters are to be sent into the environment variables. You may also hard code the variables if that suits you, but i would encourage you to use either environment variables or argparse.
```python
### Setup of S3 parameters and helper functions

#Names of the buckets
trainingbucket= os.environ['trainingbucket'] #'training'
datasetsbucket= os.environ['datasetsbucket'] #'datasets'


import boto3 #boto3 is S3 client from AWS
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
#### Prepare Dockerfile file
Now that you have your codes ready and tested locally, its time to dockerize it. Its really easy to create a Docker image, all you need is Docker installed, gather the files you want in the docker image and to create a simple file called Dockerfile. A Dockerfile is declarative, and each command is only processed after you run `Docker build`.
The following is an example of a Dockerfile
```Dockerfile
FROM tensorflow/tensorflow:nightly-gpu
ADD requirements.txt /
ADD image_classification_single.py / 
RUN apt update && \
    apt install -y  software-properties-common build-essential graphviz 
RUN pip3 install -r requirements.txt
```

Most Dockerfiles start off with a baseline image. There are a lot of images on [DockerHub](https://hub.docker.com/) and chances are that there's one that fits your purpose. Take for example, in my case i wanted to use the latest Tensorflow with GPU support. Instead of creating a setup with CUDA and go through all the installation headache, i would simply use a pre-made docker image by Tensorflow, complete with CUDA and all. So i create a `FROM` command followed by the tag `tensorflow/tensorflow:nightly-gpu`. 

Next i would want to copy all the stuff i want into the docker image. We accomplish this with the `ADD` command, followed by 2 arguments. The first argument is the path to the file, relative to the location of the Dockerfile file. The second argument is the path that i want to put inside docker. So it will look something like `ADD requirements.txt /`. 

Now, we have the files we need, but the codes won't run without the dependancies. It depends heavily on your training codes, it may require both OS and python dependancies. In this example, i only need graphviz and to install some python packages. To this end, you can use the `RUN` command. For example, i want to install all the python packges in my requirement.txt that i added earlier, i will use `RUN pip3 install -r requirements.txt`.

#### Build a docker image
So we have the 
#### Export/Save the docker image as a file
#### Prepare kubernetes job yaml file
#### Transfer to Kubernetes client

### On the Kubernetes client
#### Upload datasets onto S3
#### Load the docker image file as a docker image
#### Push the docker image to the Docker Registry
#### Run the job yaml file

## Looking forward
