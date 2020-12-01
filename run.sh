
#build the image locally
sudo docker build -t onnx-ops .

#test the image locally 
sudo docker run -ti -p 5001:5001 onnx-ops bash

#login
az login

#create a resource group 
az group create --name onnxopsdemo --location eastus

#create a container registry 
az acr create --resource-group onnxopsdemo --name onnxopscontainerreg --sku Basic

#login to container reg 
az acr login --name onnxopscontainerreg

#list images 
docker images

#tag image 

az acr list --resource-group onnxopsdemo --query "[].{acrLoginServer:loginServer}" --output table

docker tag onnx-ops:latest onnxopscontainerreg.azurecr.io/onnx-ops:latest

#push to the acr registry 

docker push onnxopscontainerreg.azurecr.io/onnx-ops:latest

#list all the images 
az acr repository list --name onnxopscontainerreg.azurecr.io --output table

############################Deploy to AKS cluster ################

### Create AKS cluster ###

az aks create --resource-group onnxopsdemo  --name onnxopscluster --node-count 2 --generate-ssh-keys --attach-acr onnxopscontainerreg

### install aks extension 
az aks install-cli


# get AKS creddential 
az aks get-credentials --resource-group onnxopsdemo --name onnxopscluster


#check status 
kubectl get nodes

#deploy the app 
kubectl apply -f onnx-app-docker-ws.yaml

#monitor the app 
kubectl get service onnxapps --watch



