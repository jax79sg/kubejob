#!/usr/bin/env python
# coding: utf-8

# # Image classification from scratch
# 
# **Author:** [fchollet](https://twitter.com/fchollet)<br>
# **Date created:** 2020/04/27<br>
# **Last modified:** 2020/04/28<br>
# **Description:** Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.

import boto3
from botocore.client import Config


### Setup of S3 parameters and helper functions
trainingbucket= os.environ['trainingbucket'] #'training'
datasetsbucket= os.environ['datasetsbucket'] #'datasets'
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


