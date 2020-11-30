FROM  python:3.6
MAINTAINER mufajjul ali mufajjul.ali@gmail.com



WORKDIR /onnxops

COPY . .


RUN apt-get -y update

RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y


RUN pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=/onnxops/webservice/mnistws.py

#ENTRYPOINT ["python3"]
#CMD ["cd", "webservice"]

WORKDIR /onnxops/webservice
CMD ["flask", "run"]

