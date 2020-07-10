FROM tensorflow/tensorflow:nightly-gpu
ADD requirements.txt /
RUN apt update && \
    apt install -y wget software-properties-common build-essential graphviz 
RUN wget https://github.com/wagoodman/bashful/releases/download/v0.0.10/bashful_0.0.10_linux_amd64.deb && \
    dpkg -i bashful_0.0.10_linux_amd64.deb
RUN pip3 install -r requirements.txt
ADD image_classification_multi.py /
ADD s3utility.py /
ADD download_datasets.py /
ADD runall.yml /
ADD runall.sh /

#CMD ["bashful", "run","/runall.yml"]

