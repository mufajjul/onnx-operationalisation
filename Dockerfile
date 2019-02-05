FROM ubuntu:latest
MAINTAINER mufajjul ali mufajjul.ali@gmail.com
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

COPY ./webservice
WORKDIR /webservice
RUN pip install -r requirements.txt
ENTRYPOINT ['python']
CMD [mnistws.py]


