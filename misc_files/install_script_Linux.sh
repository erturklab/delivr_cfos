#! /bin/bash

#install docker, adapted from here: https://docs.docker.com/engine/install/ubuntu/
#NOTE this assumes you don't have docker installed already. If it's already there, it might get overwritten! 

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release unzip bzip2

#install docker package keys 
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

#adds docker keys to apt-get keyring
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

#install docker engine 
sudo apt-get update -y
 sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

#verify that docker is installed correctly 
sudo service docker start
sudo docker run -rm hello-world

#install nvidia dockertools
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit-base
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

#download + unpack Ilastik 
#mkdir ./Ilastik
#wget -O ./Ilastik/Ilastik.tar.bz2 https://files.ilastik.org/ilastik-1.4.0rc2-Linux.tar.bz2
#tar -xf ./Ilastik/Ilastik.tar.bz2

#download + unpack FIJI 
#mkdir ./Fiji
#wget -O ./Fiji/Fiji.zip https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
#unzip ./Fiji/Fiji.zip

#download DELiVR docker image (if available, for testing purposes this is delivered with the script) 
#assuming it's already here, install it 
sudo docker load -i ./delivr_*.tar.gz




