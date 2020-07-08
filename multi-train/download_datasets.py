#!/usr/bin/env python
# coding: utf-8

# # Image classification from scratch
# 
# **Author:** [fchollet](https://twitter.com/fchollet)<br>
# **Date created:** 2020/04/27<br>
# **Last modified:** 2020/04/28<br>
# **Description:** Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.

# ## Introduction
# 
# This example shows how to do image classification from scratch, starting from JPEG
# image files on disk, without leveraging pre-trained weights or a pre-made Keras
# Application model. We demonstrate the workflow on the Kaggle Cats vs Dogs binary
#  classification dataset.
# 
# We use the `image_dataset_from_directory` utility to generate the datasets, and
# we use Keras image preprocessing layers for image standardization and data augmentation.
# 

# ## Setup
# 

from s3utility import s3_download_file
from s3utility import s3_upload_file
from s3utility import s3_upload_folder
import boto3
from botocore.client import Config
import os
import tensorflow as tf

### Setup of S3 parameters 
trainingbucket= os.environ['trainingbucket'] #'training'
datasetsbucket= os.environ['datasetsbucket'] #'datasets'
s3 = boto3.resource('s3',
                    endpoint_url= os.environ['endpoint_url'] ,
                    aws_access_key_id= os.environ['aws_access_key_id'] ,
                    aws_secret_access_key= os.environ['aws_secret_access_key'],
                    config=Config(signature_version= os.environ['signature_version']),
                    region_name= os.environ['region_name'])


import os.path

isFileExists=os.path.isfile("kagglecatsanddogs_3367a.zip")
if not isFileExists:
   s3_download_file(s3=s3, localfile='kagglecatsanddogs_3367a.zip',bucket=datasetsbucket,s3path='kagglecatsanddogs_3367a.zip')

isDirExists=os.path.isdir("PetImages")
if not isDirExists:
   from zipfile import ZipFile
   with ZipFile('kagglecatsanddogs_3367a.zip', 'r') as zipObj:
      # Extract all the contents of zip file in current directory
      zipObj.extractall()

# ### Filter out corrupted images
# 
# When working with lots of real-world image data, corrupted images are a common
# occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
#  in their header.
# 


import os

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)



