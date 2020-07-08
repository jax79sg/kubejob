#!/usr/bin/env python
# coding: utf-8

# # Image classification from scratch
# 
# **Author:** [fchollet](https://twitter.com/fchollet)<br>
# **Date created:** 2020/04/27<br>
# **Last modified:** 2020/04/28<br>
# **Description:** Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
import os
from glob import glob

def s3_download_file(s3,localfile,bucket,s3path):
    print("S3 Download s3://"+bucket+"/" + s3path + " to " + localfile )
    s3.Bucket(bucket).download_file(s3path,localfile)
    
def s3_upload_file(s3,localfile,bucket,s3path):
    print("S3 Uploading " + localfile + " to s3://"+bucket + s3path+localfile)
    s3.Bucket(bucket).upload_file(localfile,s3path+localfile)
    
def s3_upload_folder(s3,folder, bucket,s3path):
    print("Processing folder")
    for file in glob(folder+"/**/*",recursive=True):
      if (os.path.isdir(file)) == False:  
        print("Processing " + file)
        s3_upload_file(s3,bucket='training',localfile=file,s3path='')


