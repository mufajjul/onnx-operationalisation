FROM tiangolo/uwsgi-nginx-flask:python3.8

#docker container listens to for incoming request 
#ENV LISTEN_PORT 8000
#EXPOSE 9000

WORKDIR onnxapp

COPY . /onnxapp

RUN apt-get -y update

RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN pip install -r /onnxapp/requirements.txt

WORKDIR /onnxapp/app

ENV NGINX_WORKER_PROCESSES auto