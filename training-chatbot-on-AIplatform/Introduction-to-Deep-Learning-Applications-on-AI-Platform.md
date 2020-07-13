# AI Platform Advanced Developers Tutorial 1
### Training Deep Learning Algorithms using Kubernetes on the AI Platform <br> By Jadle@Slack

### Overview of Article

This article builds on top of the initial <a href=https://github.com/jax79sg/kubejob/blob/master/single-train/README.md>article</a> by Jax. In his article, the basics of building a Docker image, pushing it to a Docker repository and running the training algorithm in Kubernetes is discussed. In this article, we will be training a chatbot using <a href=https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>movie dialogue</a> data. The initial steps of building the Docker image and sending the job to Kubernetes are largely similar but in this case, it is more targeted towards the AI platform.

### Building Our Docker Image

<a href=https://kubernetes.io/>Kubernetes</a> is an orchestration tool which automates management of containerized applications. Before we discuss how to perform training of your Deep Learning algorithm using Kubernetes, your <a href=https://www.docker.com/>Docker</a> image needs to be uploaded into the Docker repository in the AI platform. This requires you to package all required software libraries and tools into a <a href=https://docs.docker.com/engine/reference/commandline/build/>Docker image</a> in an Internet-enabled machine with Docker installed built. 
```bash
docker build -t my_tensorflow:2.2.0-gpu-dialogue .
```
In this example, our <code>dockerfile</code> is relatively straightforward. _Since the docker image that we pull is running as the root user, we do not need to specify_ <code>sudo</code> _in our dockerfile_.
```dockerfile
FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update && apt-get -y upgrade
RUN pip install nltk boto3 numpy pandas pickle
COPY deep_learning_train.py /home/deep_learning_train.py
WORKDIR /home/
```

Check that your Docker image has been successfully built using <code>docker image list</code>. Assuming that the build was successful, you should see your docker image in the list of Docker images available locally. After your image is built, save via:
```bash
docker save --output my_docker_image.tar my_tensorflow:2.2.0-gpu-dialogue
```
The extracted docker image <code>my_docker_image.tar</code> will be saved in the path where you ran the command. Now, you can transfer your Docker image to the client machine before uploading it to Docker repository in the AI platform. In the client machine, run the following commands:
```bash
docker load my_docker_image.tar my_tensorflow:2.2.0-gpu-dialogue
docker tag my_tensorflow:2.2.0-gpu-dialogue dockrepo.dh.gov.sg/my_tensorflow:2.2.0-gpu-dialogue
docker push dockrepo.dh.gov.sg/my_tensorflow:2.2.0-gpu-dialogue
```
Once the Docker image is pushed into the Docker repository of the AI platform, we can proceed to use Kubernetes to train our model using the AI platform's computing resources. 

### Deploying Our Image in Kubernetes

Before calling Kubernetes, we first need to define a .yaml file. This file specifies key information for Kubernetes, including which image to pull, as well as the optimal amount of resources which you require while training your model. Do note that the pod will not run if Kubernetes is unable to assign the required resources specified, so please refrain from requesting large amount of resources. 

Let us now look at how to define a <a href=https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects/>.yaml</a> file. We remark that the number of spacing between the different levels is not important, as long as it is _consistent_.
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dialogue-train
  labels:
    purpose: dialogue-train
spec:
  containers:
  - name: dialogue-train-container
    image: dockrepo.dh.gov.sg/my_tensorflow:2.2.0-gpu-dialogue
    resources:
      limits:
        cpu: "8"
        memory: "64Gi"
        nvidia.com/gpu: "1"
      requests:
        cpu: "4"
        memory: "48Gi"
        nvidia.com/gpu: "1"
    command: ["python", "/home/dialogue_trial_transformer_tf_ver2.py"]
    args: ["-b", "128", "-d", "100", "-n", "500000"]
  restartPolicy: Never
```

We bring to your attention the following parameters. Under <code>spec</code>, the image specifies the Docker image to pull from the repository, while the <code>resources</code> indicate how much compute resources your code is requesting. One interesting thing to note is that Kubernetes is able to <a href=https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/>overwrite</a> the underlying Docker <code>ENTRYPOINT</code> and <code>CMD</code> arguments in your Docker image with the <code>command</code> and <code>args</code> settings in the .yaml file.

Start the training by going by entering the following command:
```bash
kubectl apply -f dialogue_train.yaml
```
Check that the pods statuses and ensure that they are starting or running by running <code>kubectl get pods</code>. If your code prints or displays the training progress, you can check on the progress via:
```bash
kubectl logs my-deep-learning-training-kube-1
```
<img src="dialogue_train_screenshot.JPG"></img>

You can also check whether your training has completed as well:
```bash
kubectl get pods
```
<img src="Kubernetes Get Pods.JPG"></img>
If Kubernetes is still running your training algorithm, the <code>STATUS</code> will display <code>RUNNING</code> instead.

### Details for Advanced Users

The earlier section was a basic introduction into how to deploy your deep learning algorithm into the Kubernetes cluster. However, there are intrincacies into how your training code should run while in the cluster. This is done in our example to highlight some of the best practises within the industry.

#### Loading the Data from a Central Storage

Firstly, the Docker image size could potentially increase exponentially if the <code>dockerfile</code> includes the training dataset within the image built. To mitigate this, only codes (including cloning of code repositories via <code>git clone</code>) should be packaged into the Docker image. We recommend to to store your training data in the central S3 storage or the Hadoop cluster since these components were designed to store large amounts of data.

Let us look at how to load our data from an S3 bucket. We use the <code>boto3</code> python package to connect to the S3 bucket.
```python
import boto3

# Load the file from the S3 bucket. #
s3_client = boto3.resource(
    's3',
    endpoint_url='http://minio.dsta.ai:9000' ,
    aws_access_key_id='your_user_name',
    aws_secret_access_key='your_password')
s3_bucket = s3_client.Bucket('your_bucket_name')

for obj in s3_bucket.objects.all():
    key = obj.key
    
    # Download the data. #
    if key == "data/movie_dialogue.pkl":
        tmp_pkl = obj.get()["Body"].read()
        data_tuple, idx2word, word2idx = pkl.loads(tmp_pkl)
    
    # Download all codes. #
    if key.find("codes/") != -1 and key.endswith(".py"):
        local_file_name = key.replace("codes/", "")
        s3_bucket.download_file(key, local_file_name)
```

In our example, we also wrote our custom python script (<code>tf_ver2_transformer.py</code>) to implement a slightly modified version of the <a href=https://arxiv.org/abs/1706.03762>Transformer</a> (a final residual connection connecting the embedding inputs to the outputs is applied). Using <code>s3_bucket.download_file(key, local_file_name)</code>, this script (and all other python scripts in the S3 bucket) is downloaded to the current working directory and directly imported into the main python program. The pickle file of the data <code>movie_dialogue.pkl"</code> is also read into memory using <code>obj.get()["Body"].read()</code>.

When your training runs for a long time, it is desired to save your model parameters periodically. The trained model can then be downloaded directly from the MINIO interface and deployed to another machine if desired. This is useful, for example, when there is a need to deploy a trained model on-site without access to the AI platform. 
```python
# Save the model to S3 bucket. #
if n_iter % save_s3_step == 0:
    model_files = [x[2] for x in os.walk(model_ckpt_dir)][0]
    for model_file in model_files:
        tmp_bucket_object = model_ckpt_dir + "/" + model_file
        with open(tmp_bucket_object, "rb") as tmp_model_file:
            s3_bucket.Object(tmp_bucket_object).put(Body=tmp_model_file)
```
Moving forward, we will be working on an approach to allow us to specify which training program to run as an argument to minimise building multiple Docker images which call the same building blocks. Once done, this article will be updated accordingly. 

That's it! We hope that you enjoyed reading this article :).

#### Appendix A:
The python is here.  
```python
import os
import time
import boto3
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from nltk import wordpunct_tokenize as word_tokenizer

# Define the weight update step. #
#@tf.function
def train_step(
    model, x_encode, x_decode, x_output, 
    optimizer, learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    with tf.GradientTape() as grad_tape:
        output_logits = model(x_encode, x_decode)
        
        tmp_losses = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=x_output, logits=output_logits), axis=1))
    
    tmp_gradients = \
        grad_tape.gradient(tmp_losses, model.trainable_variables)
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(tmp_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model.trainable_variables))
    return tmp_losses

# Parse the arguments. #
parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", "--batch", help="Batch size", required=True)
parser.add_argument(
    "-d", "--display", help="Display steps", required=True)
parser.add_argument(
    "-n", "--n_iterations", help="Number of iterations", required=True)

args = parser.parse_args()
print("Batch Size:", str(args.batch))
print("Display steps:", str(args.display))
print("No. of iterations:", str(args.n_iterations))

# Load the file from the S3 bucket. #
s3_client = boto3.resource(
    's3',
    endpoint_url='http://minio.dsta.ai:9000' ,
    aws_access_key_id='your_user_name',
    aws_secret_access_key='your_password')
s3_bucket = s3_client.Bucket('your_S3_bucket')

for obj in s3_bucket.objects.all():
    key = obj.key
    
    # Download the data. #
    if key == "data/movie_dialogue.pkl":
        tmp_pkl = obj.get()["Body"].read()
        data_tuple, idx2word, word2idx = pkl.loads(tmp_pkl)
    
    # Download all codes. #
    if key.find("codes/") != -1 and key.endswith(".py"):
        local_file_name = key.replace("codes/", "")
        s3_bucket.download_file(key, local_file_name)

# The custom module can only be imported after downloading the   #
# python scripts from the Minio server to the current directory. #
import tf_ver2_transformer as tf_transformer

num_data  = len(data_tuple)
vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size))

SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Model Parameters. #
batch_size = int(args.batch)
seq_encode = 15
seq_decode = 16

num_layers  = 6
num_heads   = 16
prob_keep   = 0.9
hidden_size = 1024
ffwd_size   = 4*hidden_size

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = int(args.n_iterations)
restore_flag  = False
display_step  = int(args.display)
save_s3_step  = 10000
cooling_step  = 1000
warmup_steps  = 2500
anneal_step   = 2000
anneal_rate   = 0.75

model_ckpt_dir  = \
    "TF_Models/transformer_seq2seq"
train_loss_file = "dialogue_train_loss_transformer.csv"

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

print("Building the Transformer Model.")
start_time = time.time()

seq2seq_model = tf_transformer.TransformerNetwork(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size, vocab_size, seq_encode, seq_decode, 
    embed_size=hidden_size, p_keep=prob_keep)
seq2seq_optimizer = tf.keras.optimizers.Adam()

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optimizer=seq2seq_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    # Download all model outputs. #
    for obj in s3_bucket.objects.all():
        key = obj.key
        if key.find(model_ckpt_dir) != -1:
            s3_bucket.download_file(key, key)
    
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros([batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros([batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)
tmp_test_dec = SOS_token * np.ones([1, seq_decode], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Transformer Network", 
      "(" + str(n_iter), "iterations).")
    
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_input[:, :]   = PAD_token
    tmp_seq_out[:, :] = PAD_token
    tmp_seq_out[:, 0] = SOS_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_tok = data_tuple[tmp_index][0].split(" ")
        tmp_o_tok = data_tuple[tmp_index][1].split(" ")

        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
        tmp_o_idx = [word2idx.get(x, UNK_token) for x in tmp_o_tok]
        
        n_input  = len(tmp_i_idx)
        n_output = len(tmp_o_idx)
        n_decode = n_output + 1

        tmp_input[n_index, :n_input] = tmp_i_idx
        tmp_seq_out[n_index, 1:n_decode] = tmp_o_idx
        tmp_seq_out[n_index, n_decode] = EOS_token

    tmp_decode = tmp_seq_out[:, :-1]
    tmp_output = tmp_seq_out[:, 1:]
    
    tmp_loss = train_step(
        seq2seq_model, tmp_input, tmp_decode, tmp_output, 
        seq2seq_optimizer, learning_rate=learn_rate_val)
    
    n_iter += 1
    tot_loss += tmp_loss.numpy()
    ckpt.step.assign_add(1)

    if n_iter % display_step == 0:
        end_time = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_time - start_tm) / 60
        start_tm   = time.time()

        tmp_test_in[:, :] = PAD_token
        sample_id = np.random.choice(num_data, size=1)
        tmp_data  = data_tuple[sample_id[0]]

        tmp_i_tok = tmp_data[0].split(" ")
        tmp_o_tok = tmp_data[1].split(" ")
        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]

        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        gen_ids = seq2seq_model.infer(tmp_test_in, tmp_test_dec)
        gen_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
        gen_phrase = " ".join(gen_phrase)

        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Batch Size:", str(batch_size) + ".")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(seq2seq_optimizer.lr.numpy()))

        print("")
        print("Input Phrase:")
        print(" ".join([idx2word[x] for x in tmp_i_idx]))
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_data[1])
        print("")
        
        # Save the training progress. #
        train_loss_list.append((n_iter, avg_loss))
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        train_loss_df.to_csv(train_loss_file, index=False)
        s3_bucket.Object("data/dialogue_train_loss.csv").put(Body=train_loss_file)
        
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        # Save the model to S3 bucket. #
        if n_iter % save_s3_step == 0:
            model_files = [x[2] for x in os.walk(model_ckpt_dir)][0]
            for model_file in model_files:
                tmp_bucket_object = model_ckpt_dir + "/" + model_file
                with open(tmp_bucket_object, "rb") as tmp_model_file:
                    s3_bucket.Object(tmp_bucket_object).put(Body=tmp_model_file)
        print("-" * 50)
```
