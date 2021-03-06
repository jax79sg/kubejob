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
import argparse
from s3utility import s3_download_file
from s3utility import s3_upload_file
from s3utility import s3_upload_folder
## Load hyperparam from argparse

argparser = argparse.ArgumentParser(description="Hyperparameters setup")
argparser.add_argument('--expid',type=str)
argparser.add_argument('--batch_size',type=int)
argparser.add_argument('--image_size_h',type=int)
argparser.add_argument('--image_size_w',type=int)
argparser.add_argument('--buffer_size',type=int)
argparser.add_argument('--dropout',type=float)
argparser.add_argument('--epochs',type=int)
argparser.add_argument('--learning_rate',type=float)

args = argparser.parse_args()

import boto3
from botocore.client import Config
import os

### Setup of S3 parameters 
trainingbucket= os.environ['trainingbucket'] #'training'
datasetsbucket= os.environ['datasetsbucket'] #'datasets'
s3 = boto3.resource('s3',
                    endpoint_url= os.environ['endpoint_url'] ,
                    aws_access_key_id= os.environ['aws_access_key_id'] ,
                    aws_secret_access_key= os.environ['aws_secret_access_key'],
                    config=Config(signature_version= os.environ['signature_version']),
                    region_name= os.environ['region_name'])


## Model Training hyperparameters
expid=args.expid
batch_size=args.batch_size
image_size_h=args.image_size_h
image_size_w=args.image_size_w
buffer_size=args.buffer_size
dropout=args.dropout
epochs=args.epochs
learning_rate=args.learning_rate



# ## Generate a `Dataset`
# 


image_size = (image_size_h, image_size_w)
#batch_size = 32

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

train_ds = train_ds.prefetch(buffer_size=buffer_size)
val_ds = val_ds.prefetch(buffer_size=buffer_size)


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

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)


# ## Train the model
# 


#epochs = 2
callbacks = [
    keras.callbacks.ModelCheckpoint(expid+"_catdogclassification_save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history=model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
model.save(expid+"_catdogclassification_model") 


### Up until this point,allthe model files are saved on container. After this container finishes execution, the files will be gone.
### Start saving the checkpoints and model files.
import json
with open(expid+'_catdogclassification.json', 'w') as fp:
    json.dump(history.history, fp)
s3_upload_file(s3=s3,bucket=trainingbucket,localfile=expid+'_catdogclassification.json',s3path='')

s3_upload_folder(s3=s3, bucket=trainingbucket,folder=expid+'_catdogclassification_model',s3path='')

for epochrun in range(epochs):
    s3_upload_file(s3=s3, bucket=trainingbucket,localfile=expid+'_catdogclassification_save_at_'+str(epochrun+1)+'.h5',s3path='')


# We get to ~96% validation accuracy after training for 50 epochs on the full dataset.

