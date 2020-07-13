# Training a Open Source ML/DL model on AI Platform (Kubernetes) - Part I.

|   |   |
| ------------- | ------------- |
| **Difficulty**  | ![Normal](../normal40.png)  |
| **Contact**  | Jax@[Slack](https://aidevplatform.slack.com) |

For a lot of us, we have been building and/or training our models on our own GPU enabled machines. Our single GPU enabled machines at best can cater up to 11GB of RAM, while this is sufficient for smaller models, its a challenge if we want to train larger models from scratch (E.g. BERT). Other than the scale, the speed of training on Ti1080s and RTX2080s are limited, so moving the training onto Kubernetes where V100 GPUs are available will significantly improve the above.

## Who should try this?
### 3rd Parties
If you have a Deep Learning architecture you got from someone or pulled from open sourced research, and you need to perform some form of training on the model without intimate knowledge of the codes, this method would be most suitable for you. 
### Development in Docker
For those who are regularly developing their codes in Docker, this would be very apt for them as well. The advantage of developing DL models in Docker is that they highly flexible when it comes to using different frameworks and versions. For example, you don't have to crack your head on different versions of CUDA on your machine, just make sure you have a docker for every version. Most times, you don't even have to worry about this as the frameworks such as Tensorflow come with their own docker images anyway.

## What will be achieved at the end of this article?
This example uses a 3rd party end to end image classification code. The code is customised to download dataset frmo S3 onto itself and also to upload model checkpoints and final results onto S3 after training. By the end of this article, you will get acquainted with very basic use of Docker and Kubernetes. You would be able to submit jobs to Kubernetes and get the results from S3 object stores.

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
5. Transfer to Kubernetes client

### On the Kubernetes client
0. Load datasets onto S3
1. Load the docker image file as a docker image
2. Push the docker image to the Docker Registry
3. Prepare kubernetes job yaml file
4. Run the job yaml file

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

A sample of a training code is found in [image_classification_single.py](https://raw.githubusercontent.com/jax79sg/kubejob/master/single-train/image_classification_single.py). The only change to this code to the original is such that the zipped datasets would be downloaded from S3 and then extracted for processing, after training, the model and results are saved in S3. Your situation could be different, please exercise your own considerations. 

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
Now that you have your codes ready and tested locally, its time to dockerize it. Its really easy to create a Docker image, all you need is Docker installed, gather the files you want in the docker image and to create a simple file called Dockerfile. A Dockerfile is declarative, and the commands are only processed after you run `docker build`.
The following is the Dockerfile for this example.
```Dockerfile
FROM tensorflow/tensorflow:nightly-gpu
ADD requirements.txt /
ADD image_classification_single.py / 
RUN apt update && \
    apt install -y  software-properties-common build-essential graphviz 
RUN pip3 install -r requirements.txt
```
Most Dockerfiles start off with a baseline image. There are a lot of images on [DockerHub](https://hub.docker.com/) and chances are that there's one that fits your purpose. Take for example, in this case the latest Tensorflow with GPU support is desired. Instead of creating a setup with CUDA and go through all the installation headache, a pre-made docker image by Tensorflow complete with CUDA and all is used instead. To do this, a `FROM` command followed by the tag `tensorflow/tensorflow:nightly-gpu` is used. 

Next, copy all the stuff required into the docker image by using the `ADD` command, followed by 2 arguments. The first argument is the path to the file, relative to the location of the Dockerfile file. The second argument is the path inside the docker image (The folders will be created automatically if it doesn't exists). So it will look something like `ADD requirements.txt /`. 

The codes won't run without the dependancies. In this example, graphviz and some python packages are quired. To this end, you can use the `RUN` command. For this  example, use `RUN pip3 install -r requirements.txt`. After this is acheived, you may proceed to build the image.

#### Build a docker image
To build the image with the docker file, you need to run the following command in the same folder where Dockerfile is located.
In this example, under the `kubejob/single-train` folder.
```bash
jax@getafix: docker build .t image-classification-single
Removing intermediate container d941290dff33
 ---> 3793f6e38a2f
Successfully built 3793f6e38a2f
Successfully tagged image-classification-single:latest
```
You can run `docker images` and see the docker image listed.
```bash
jax@getafix: docker images
REPOSITORY                                        TAG                              IMAGE ID            CREATED             SIZE
image-classification-single                       latest                           3793f6e38a2f        2 minutes ago       3.49GB
```
At this point, you can run the docker image on your own computer and run the training. This is the closest to which how it will run on kubernetes. Successfullying running this step will ensure that your image will most likely run properly on kubernetes.
```bash
docker run -it --gpus all --env-file env.list image-classification-single python3 /image_classification_single.py
```
`--gpus all` directs docker to use the GPU (provided nvidia-docker is installed)

`--env-file env.list` loads the environment variables (S3 parameters) into the docker container.

`python3 /image_classification_single.py` is the command to run your training script

When you run the command, you would see something like following (*Note that this script ensures only 1 epoch is run for testing sake*)
```bash
2020-07-11 13:45:41.014731: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2.4.0-dev20200705
S3 Download s3://datasets/kagglecatsanddogs_3367a.zip to kagglecatsanddogs_3367a.zip
Deleted 1590 images
Found 23410 files belonging to 2 classes.
Using 18728 files for training.
2020-07-11 13:45:51.001862: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-07-11 13:45:51.005333: E tensorflow/stream_executor/cuda/cuda_driver.cc:314] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2020-07-11 13:45:51.005350: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: 7c20885796ae
2020-07-11 13:45:51.005355: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: 7c20885796ae
2020-07-11 13:45:51.005777: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:200] libcuda reported version is: 440.100.0
2020-07-11 13:45:51.005818: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:204] kernel reported version is: 440.100.0
2020-07-11 13:45:51.005825: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:310] kernel version seems to match DSO: 440.100.0
2020-07-11 13:45:51.006963: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-07-11 13:45:51.036899: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2592000000 Hz
2020-07-11 13:45:51.038416: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x438ea80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 13:45:51.038456: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Found 23410 files belonging to 2 classes.
Using 4682 files for validation.
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9
108/293 [==========>...................] - ETA: 22s - loss: 0.6661 - accuracy: 0.6215Corrupt JPEG data: 228 extraneous bytes before marker 0xd9
119/293 [===========>..................] - ETA: 20s - loss: 0.6637 - accuracy: 0.6241Warning: unknown JFIF revision number 0.00
149/293 [==============>...............] - ETA: 17s - loss: 0.6568 - accuracy: 0.6288Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
154/293 [==============>...............] - ETA: 16s - loss: 0.6547 - accuracy: 0.6313Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
160/293 [===============>..............] - ETA: 15s - loss: 0.6526 - accuracy: 0.6335Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
163/293 [===============>..............] - ETA: 15s - loss: 0.6526 - accuracy: 0.6342Corrupt JPEG data: 239 extraneous bytes before marker 0xd9
293/293 [==============================] - ETA: 0s - loss: 0.6259 - accuracy: 0.6564Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9
293/293 [==============================] - 39s 133ms/step - loss: 0.6259 - accuracy: 0.6564 - val_loss: 0.6948 - val_accuracy: 0.5043
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
2020-07-11 13:46:35.042243: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
S3 Uploading catdogclassification.json to s3://trainingcatdogclassification.json
Processing folder
Processing catdogclassification_model/saved_model.pb
S3 Uploading catdogclassification_model/saved_model.pb to s3://trainingcatdogclassification_model/saved_model.pb
Processing catdogclassification_model/variables/variables.data-00000-of-00001
S3 Uploading catdogclassification_model/variables/variables.data-00000-of-00001 to s3://trainingcatdogclassification_model/variables/variables.data-00000-of-00001
Processing catdogclassification_model/variables/variables.index
S3 Uploading catdogclassification_model/variables/variables.index to s3://trainingcatdogclassification_model/variables/variables.index
S3 Uploading catdogclassification_save_at_1.h5 to s3://trainingcatdogclassification_save_at_1.h5
```
#### Export/Save the docker image as a file
The above steps ensured that you have a running training script that will work on Docker. The next step is to export the docker image so you can transfer it to the Kubernetes client.
```bash
jax@getafix: docker save image-classification-single -o image-classification-single.tar
```
Now transfer the docker image to Kubernetes client

### On the Kubernetes client
#### Upload datasets onto S3
On the Kubernetes client, a MINIO client (commandline) has been configured for you to manage your buckets. 
In this example, the following command would have been executed.
```bash
jax@getafix: /home/user/mc mb myminio/datasets
Bucket created successfully `myminio/datasets`.

jax@getafix: /home/user/mc cp kagglecatsanddogs_3367a.zip myminio/datasets/kagglecatsanddogs_3367a.zip
...atsanddogs_3367a.zip:  786.68 MiB / 786.68 MiB ┃▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃ 55.17 MiB/s 14s
```
Some basic usage of the commmands are as follows.

To create a new bucket 'mynewbucket', run the following command.
```bash
/home/user/mc mb myminio/mynewbucket
```
To upload a folders or files 'mylocalfolderorfile', run the following command.
```bash
/home/user/mc cp mylocalfolderorfile myminio/mynewbucket/
```
To download a folder or file 'mynewbucket', run the following command.
```bash
/home/user/mc cp myminio/mynewbucket/myremotefolderorfile mylocalfolderorfile
```
#### Push the docker image to the Docker Registry
The docker image file that we copied over from our own computer needs to be loaded into the Docker Registry on the AI Platform.
```bash
#Loads the tar file (docker image) into the client's local docker repo.
jax@getafix: docker load -i image-classification-single.tar 
Loaded image: image-classification-single:latest

#Tag the uploaded image to bear the url to the AI Platorm's Docker Rgistry.
jax@getafix: docker tag image-classification-single dockrepo.dh.gov.sg:5000/image-classification-single:latest 

#Send the image from local Docker to the AI Platform's docker registry.
jax@getafix: docker push dockrepo.dh.gov.sg/image-classification-single:latest 
The push refers to repository [myregistry.com:5000/image-classification-single]
94f43a58fd54: Layer already exists 
c394cd29e2f8: Layer already exists 
a7aabfd17751: Layer already exists 
be0113cc7bc0: Layer already exists 
21f7133a99fb: Layer already exists 
626976cc3d82: Layer already exists 
63beefd08b72: Layer already exists 
e8f3214614e5: Layer already exists 
cc3fc5898d66: Layer already exists 
7db070456ae6: Layer already exists 
10a49ffdc6d4: Layer already exists 
45a3946bc76a: Layer already exists 
43895ac43b99: Layer already exists 
808fd332a58a: Layer already exists 
b16af11cbf29: Layer already exists 
37b9a4b22186: Layer already exists 
e0b3afb09dc3: Layer already exists 
6c01b5a53aac: Layer already exists 
2c6ac8e5063e: Layer already exists 
cc967c529ced: Layer already exists 
latest: digest: sha256:1df82e72ddb603195af7b57034d536190ccbc2c3ee59faed9a4844d3c079b8da size: 4515
```
#### Prepare kubernetes job yaml file
The final step of preparation is to create a Kubernetes yaml file.
```bash
apiVersion: v1
kind: Pod
metadata:
   name: single-train
spec:
         containers:
         - name: test-image-classification-single
           image: dhrepo.dh.gov.sg:5000/image-classification-single"
           env:
           - name: trainingbucket
             value: training
           - name: datasetsbucket
             value: datasets
           - name: endpoint_url
             value: http://minio.dsta.ai:9001
           - name: aws_access_key_id
             value: user
           - name: aws_secret_access_key
             value: password
           - name: signature_version
             value: s3v4
           - name: region_name
             value: us-east-1
           resources:
              requests:
                 cpu: "2"
                 memory: "2Gi"
           command: ["python3","/image_classification_single.py"]
         restartPolicy: Never
```
The above yml is a minimal yaml required for this example, with the important ones stated below.

`image` - Specify the image of that the job will run.

`env` - List the environment variables required to pass to the container.

`requests` - Minimum resources required for this container to run

`command` - command to run the script

#### Run the pod yaml file
Lastly, run the pod submission.
```bash
jax@getafix: kubectl apply -f kube-single.yml
pod/single-train created
```
When a kubernetes pod has been successfully submited, you can monitor 2 things, as indicated below.
Following command will display the pod that are in Kubernetes. The pod name takes after the `meta-data > name` in the yaml file.
```bash
jax@getafix: kubectl get pods
NAME                 READY   STATUS      RESTARTS   AGE
single-train         0/1     Completed   0          10h

```
The following command will display the running stdout of the pod that you just created. The output should be identical to the docker run output above.
```bash
jax@getafix: kubectl logs single-train
2020-07-13 03:22:24.361047: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-07-13 03:22:35.524867: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-07-13 03:22:35.524895: E tensorflow/stream_executor/cuda/cuda_driver.cc:314] failed call to cuInit: UNKNOWN ERROR (-1)
2020-07-13 03:22:35.524910: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (single-train-94nvt): /proc/driver/nvidia/version does not exist
2020-07-13 03:22:35.525161: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-07-13 03:22:35.529438: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2592000000 Hz
2020-07-13 03:22:35.529656: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x43dac60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-13 03:22:35.529674: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9
2.4.0-dev20200705
S3 Download s3://datasets/kagglecatsanddogs_3367a.zip to kagglecatsanddogs_3367a.zip
Deleted 1590 images
Found 23410 files belonging to 2 classes.
Using 18728 files for training.
Found 23410 files belonging to 2 classes.
Using 4682 files for validation.
Corrupt JPEG data: 228 extraneous bytes before marker 0xd9
 95/293 [========>.....................] - ETWarning: unknown JFIF revision number 0.00
Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9
287/293 [============================>.] - ETA: 1s - Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9
293/293 [==============================] - 91s 309ms/step - loss: 0.6408 - accuracy: 0.6454 - val_loss: 0.6949 - val_accuracy: 0.4957
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
2020-07-13 03:24:11.022698: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
S3 Uploading catdogclassification.json to s3://trainingcatdogclassification.json
Processing folder
Processing catdogclassification_model/saved_model.pb
S3 Uploading catdogclassification_model/saved_model.pb to s3://trainingcatdogclassification_model/saved_model.pb
Processing catdogclassification_model/variables/variables.data-00000-of-00001
S3 Uploading catdogclassification_model/variables/variables.data-00000-of-00001 to s3://trainingcatdogclassification_model/variables/variables.data-00000-of-00001
Processing catdogclassification_model/variables/variables.index
S3 Uploading catdogclassification_model/variables/variables.index to s3://trainingcatdogclassification_model/variables/variables.index
S3 Uploading catdogclassification_save_at_1.h5 to s3://trainingcatdogclassification_save_at_1.h5
```

## What's next
The above is a very simple example to demonstrate the use of Docker and Kubernetes. However, running a single training on a 32GB V100 GPU card is not efficient and running as a pod directly is not typical. The next article [Part II](https://github.com/jax79sg/kubejob/multi-train/README.md) demonstrates how this same example can be enhanced to support some form of hyperparameter tuning (E.g. Running several training jobs with different hyperparamters concurrently, as long as the total GPU ram is not exceeded).

## Call for contribution
The above example is one of many possible ways to utilise Kubernetes for our AI development. If you have an interesting idea, please feel free to share it on our Slack page.
