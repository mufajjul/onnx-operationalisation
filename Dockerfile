FROM python:3

MAINTAINER mufajjul ali mufajjul.ali@gmail.com

#RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential
#RUN pip install --upgrade pip

WORKDIR /webservice
COPY . /webservice

WORKDIR /mlmodel
copy . /mlmodel


RUN pip install -r requirements.txt

#CMD [ "export", "FLASK_APP=mnistws.py" ]
#WORKDIR /webservice
ENTRYPOINT ["python"]
CMD ["mnistws.py" ]


#CMD [mnistws.py]


