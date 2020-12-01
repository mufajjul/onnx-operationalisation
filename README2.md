# onnx-deployment
In this section, we are going to discuss how to containerize an ONNX model, and then deploy it on an AKS cluster in Azure.  

## onnx-containerization
We are going to use Docker to containerize a Flask Web service that exposes a model as a REST endpoint.  To learn more about Docker, please visit the following link (https://www.docker.com/)

First, you need to install Docker on your local machine. You can download it from the following link (https://www.docker.com/)
 
### Create a Dockerfile 
A Docker file contains a list of definitions for your image. Things like,  dependencies, libraries, environment variables, etc., are described in the definition file.  Below is an example of a Dockerfile we created. 
 

```python
#base Docker image
FROM tiangolo/uwsgi-nginx-flask:python3.8

#define working directory
WORKDIR onnxapp
COPY . /onnxapp

#install OS level dependencies
RUN apt-get -y update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

#Install python dependencies
RUN pip install -r /onnxapp/requirements.txt

#current working directory to run the app
WORKDIR /onnxapp/app
ENV NGINX_WORKER_PROCESSES auto
```

It is recommended to use a base Image that contains most of your component definitions.  In this example, we are going to use the uwsgi-nginx-flask base image, with uWSGI and Nginx for Flask base application, plus it also comes with Python 3.8. 

The image also installs all the python dependencies, such as the ONNX runtime, image/data transformation libraries and others. 

To build the Docker image, type the following:

```python
sudo docker build -t onnx-ops:latest .

```
This will build an Image called onnx-ops:latest. To see all the built images, type the following:

```python 
docker images 
```

### Test Docker image locally 

To test the Docker Image locally, we are going to create a Docker compose file. A docker compose file contains definitions of how various services will interact with others, and provides all the configuration details.  

```yaml 
services:
  web:
    image: onnx-ops:latest 
    container_name: onnx-ops   
    ports:
      - "8080:8080"
    environment:
      - FLASK_APP=main.py
      - FLASK_DEBUG=1
      - 'RUN=flask run --host=0.0.0.0 --port=8080'
    command: flask run --host=0.0.0.0 --port=808 
```

It is recommended to test the application locally first before deploying it to the Cloud. Type the following command to run it locally. 

```python 
docker-compose up -d
```

### Create a container registry 
Now we are going to create a container registry in Azure. We are going to Azure CLI.

First, you need to login to Azure. 

```python 
docker images 
```
Once you have successfully logged into Azure, you are going to create a resource group. 

```python 
az group create --name onnxopsdemo --location eastus
```

Next you are going to log into the registry.
```python 
az acr login --name onnxopscontainerreg
```
Now, we are going to prepare the Image to be uploaded to the container registry by tagging it to the Docker Image naming convention.  

```python 
docker tag onnx-ops:latest onnxopscontainerreg.azurecr.io/onnx-ops:latest
```
Now that the image is labelled correctly, we are going to push it to the container registry. 

```python 
docker push onnxopscontainerreg.azurecr.io/onnx-ops:lates
```
To view all the images in the container registry including the one we have just uploaded, type the following.

```python 
az acr repository list --name onnxopscontainerreg.azurecr.io --output table
```
The docker image is now ready to be deployed in an AKS cluster.

## AKS Deployment
We are going to use the CLI to create the AKS cluster. It is recommended to use a 3-node cluster for the production, and a one node for the dev/test environment.  

### Create a Azure Kubernaties Cluster

First, let's install the CLI extension for AKS.

```python
az aks install-cl
```
Now that we have installed the CLI extension, it is time to create a 3-node AKS cluster. Please note, this stage can take upto 10min, so be patient.

```python
az aks create --resource-group onnxopsdemo  --name onnxopscluster --node-count 3 -generate-ssh-keys --attach-acr onnxopscontainerreg
```
Next we are going to get the AKS credentials, to use the kubectl tool.

```python
az aks get-credentials --resource-group onnxopsdemo --name onnxopscluster
```
We are now going to check for the status of the deployed nodes.  

```python
kubectl get nodes
```
Before you can deploy your cluster, you need to create the provision script, where you describe how the application should be provisioned, as well as the load balancer.

```yaml
#application to be deployed
piVersion: apps/v1
kind: Deployment
metadata:
  name: onnxapps
spec:
  replicas: 1
  selector:
    matchLabels:
      app: onnxapps
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  minReadySeconds: 5 
  template:
    metadata:
      labels:
        app: onnxapps
    spec:
      nodeSelector:
        "beta.kubernetes.io/os": linux
      containers:
      - name: onnxapps
        image: onnxopscontainerreg.azurecr.io/onnx-ops:latest 
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 250m
          limits:
            cpu: 500m
---
#load balancer to be deployed
apiVersion: v1
kind: Service
metadata:
  name: onnxapps
spec:
  type: LoadBalancer
  ports:
  - port: 80
    protocol: TCP
    targetPort: 8080
  selector:
    app: onnxapps   

```

Now, you are ready to deploy your application (using the YAML file) to the AKS cluster. 

```python
kubectl apply -f onnx-app-docker-ws.yaml
```
This will deploy one service and one application to the AKS.  You can monitor the status of the deployment by typing the following command.  

```python
kubectl get service onnxapps --watch
```
### Conclusion 

In this section, we have demonstrated how to build and deploy a Docker image that contains a Restful AML Webserice in AKS.  Couple things to note, first, for production deployment, you should use a WSGI server. Secondly, you should consider using HTTPS/SSL for securing the REST endpoint, and finally think about how you are going to monitor your deployed application.

While this approach gives you the most flexibility, it is prone to errors and requires a lot of effort in configuration and gluing things together,  which can be time consuming.  I would strongly urge to consider looking at tools like Azure Machine Learning, which provides a lot of these capabilities out of the box, plus enterprise security with SLA. It will greatly improve your productivity, and would be a lot less painful. 

Azure Machine Learning (https://azure.microsoft.com/en-gb/services/machine-learning/)























