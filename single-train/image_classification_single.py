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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__) 
import os

# ## Load the data: the Cats vs Dogs dataset from S3
# 

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


import os.path

isFileExists=os.path.isfile("kagglecatsanddogs_3367a.zip")
if not isFileExists:
    s3_download_file(localfile='kagglecatsanddogs_3367a.zip',bucket=datasetsbucket,s3path='kagglecatsanddogs_3367a.zip')
  
from zipfile import ZipFile
with ZipFile('kagglecatsanddogs_3367a.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

#get_ipython().system('unzip -qq -o kagglecatsanddogs_3367a.zip')

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


# ## Generate a `Dataset`
# 


image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# ## Using image data augmentation
# 
# When you don't have a large image dataset, it's a good practice to artificially
# introduce sample diversity by applying random yet realistic transformations to the
# training images, such as random horizontal flipping or small random rotations. This
# helps expose the model to different aspects of the training data while slowing down
#  overfitting.
# 


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

# ## Standardizing the data
# 
# Our image are already in a standard size (180x180), as they are being yielded as
# contiguous `float32` batches by our dataset. However, their RGB channel values are in
#  the `[0, 255]` range. This is not ideal for a neural network;
# in general you should seek to make your input values small. Here, we will
# standardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of
#  our model.
# 

# ## Two options to preprocess the data
# 
# There are two ways you could be using the `data_augmentation` preprocessor:
# 
# **Option 1: Make it part of the model**, like this:
# 
# ```python
# inputs = keras.Input(shape=input_shape)
# x = data_augmentation(inputs)
# x = layers.experimental.preprocessing.Rescaling(1./255)(x)
# ...  # Rest of the model
# ```
# 
# With this option, your data augmentation will happen *on device*, synchronously
# with the rest of the model execution, meaning that it will benefit from GPU
#  acceleration.
# 
# Note that data augmentation is inactive at test time, so the input samples will only be
#  augmented during `fit()`, not when calling `evaluate()` or `predict()`.
# 
# If you're training on GPU, this is the better option.
# 
# **Option 2: apply it to the dataset**, so as to obtain a dataset that yields batches of
#  augmented images, like this:
# 
# ```python
# augmented_train_ds = train_ds.map(
#   lambda x, y: (data_augmentation(x, training=True), y))
# ```
# 
# With this option, your data augmentation will happen **on CPU**, asynchrously, and will
#  be buffered before going into the model.
# 
# If you're training on CPU, this is the better option, since it makes data augmentation
#  asynchronous and non-blocking.
# 
# In our case, we'll go with the first option.
# 

# ## Configure the dataset for performance
# 
# Let's make sure to use buffered prefetching so we can yield data from disk without
#  having I/O becoming blocking:
# 

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


# ## Build a model
# 
# We'll build a small version of the Xception network. We haven't particularly tried to
# optimize the architecture; if you want to do a systematic search for the best model
#  configuration, consider using
# [Keras Tuner](https://github.com/keras-team/keras-tuner).
# 
# Note that:
# 
# - We start the model with the `data_augmentation` preprocessor, followed by a
#  `Rescaling` layer.
# - We include a `Dropout` layer before the final classification layer.
# 



def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)


# ## Train the model
# 


epochs = 2
callbacks = [
    keras.callbacks.ModelCheckpoint("catdogclassification_save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history=model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
model.save("catdogclassification_model") 


### Up until this point,allthe model files are saved on container. After this container finishes execution, the files will be gone.
### Start saving the checkpoints and model files.
import json
with open('catdogclassification.json', 'w') as fp:
    json.dump(history.history, fp)
s3_upload_file(bucket=trainingbucket,localfile='catdogclassification.json',s3path='')

s3_upload_folder(bucket=trainingbucket,folder='catdogclassification_model',s3path='')

for epochrun in range(epochs):
    s3_upload_file(bucket=trainingbucket,localfile='catdogclassification_save_at_'+str(epochrun+1)+'.h5',s3path='')


# We get to ~96% validation accuracy after training for 50 epochs on the full dataset.

