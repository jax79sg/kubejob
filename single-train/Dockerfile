FROM tensorflow/tensorflow:nightly-gpu
ADD requirements.txt /
ADD image_classification_single.py / 
RUN apt update && \
    apt install -y  software-properties-common build-essential graphviz 
RUN pip3 install -r requirements.txt
#EXPOSE 8888
#CMD ["jupyter", "notebook","--ip=0.0.0.0","--allow-root"]
#CMD ["python3","/image_classification_single.py"]

