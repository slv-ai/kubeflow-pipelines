## 1. install docker and minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

minikube version

minikube start
minikube delete
####Start Minikube with the Docker driver
minikube start --driver=docker
minikube status
### 2. install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

kubectl version --client

#check kubernetes run in docker container
docker ps

kubectl get nodes
kubectl cluster-info
### 3.pip install kfp
### 4.kubeflow install on docker container
export PIPELINE_VERSION=2.4.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
### 5. install everything in a kubeflow namespace
kubectl get all -n kubeflow
#wait until all containers ready
### 6.verify the kubeflow pipelines UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
### Recommendation
EC2 Instance	t3.xlarge (4 vCPU, 16 GB RAM)
EBS Volume	80 GB gp3 SSD
OS	Ubuntu 22.04 LTS
