# python image
#FROM python:3.10.6

#ubuntu 
#FROM ubuntu:22.04
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN useradd -ms /bin/bash delivr

#install updates
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nano libtiff5 libgomp1 -y
RUN apt-get install build-essential -y
RUN apt-get install python3 python3-pip -y

# create working directory
RUN mkdir -p /delivr

#USER delivr

COPY ./delivr-cfos/ ./delivr/ 
#COPY ./delivr_cfos/requirements.txt /delivr/requirements.txt
#COPY ./delivr_cfos/setup.py /delivr/setup.py
#COPY ./delivr_cfos/README.md /delivr/README.md
WORKDIR /delivr/

# install python packages
RUN pip install -r requirements.txt
RUN pip install more-itertools 
RUN pip install imagecodecs==2023.3.16
RUN pip install numpy==1.24.4
RUN pip install matplotlib
#RUN pip install --upgrade numpy
#RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
#RUN pip install .

ENV PYTHONPATH "${PYTHONPATH}:/delivr"


# copy dictories and files 
#COPY ./delivr_cfos/downsample/ /delivr/downsample/
#COPY ./delivr_cfos/inference/ /delivr/inference/
#COPY ./delivr_cfos/models/ /delivr/models/
#COPY ./delivr_cfos/*.py . 
#COPY *.json . 


#copy external programs 
#Ilastik: https://www.ilastik.org/download.html
COPY ./ilastik-1.4.0b8-Linux/ ./ilastik/
#terastitcher: https://github.com/abria/TeraStitcher/wiki/Binary-packages#terastitcher-portable-no-gui--only-command-line-tools
COPY ./terastitcher/TeraStitcher-portable-1.11.10-Linux/ ./teraconverter/

#mBrainAligner: https://github.com/Vaa3D/vaa3d_tools/tree/master/hackathon/mBrainAligner
# Don't forget to download the libraries (lib.tar.xz) and copy them into the mbrainaligner folder: Linux libraries via the google drive link at https://github.com/Vaa3D/vaa3d_tools/tree/master/hackathon/mBrainAligner
COPY ./2023-01-18_mbrainaligner/ ./mbrainaligner/
#mbrainaligner: unpack the libraries and copy to /usr/lib
RUN tar xvJf ./mbrainaligner/lib.tar.xz -C ./mbrainaligner/ && cp ./mbrainaligner/lib/* /usr/lib
#do the same for the libraries for swc_registration 
RUN tar xvJf ./mbrainaligner/examples/swc_registration/binary/linux_bin/lib.tar.xz -C ./mbrainaligner/examples/swc_registration/binary/linux_bin/ && cp ./mbrainaligner/examples/swc_registration/binary/linux_bin/lib/* /usr/lib
#make the mbrainaligner files executable 
RUN chmod +x ./mbrainaligner/examples/swc_registration/binary/linux_bin/swc_registration
RUN chmod +x ./mbrainaligner/binary/linux_bin/global_registration
RUN chmod +x ./mbrainaligner/binary/linux_bin/local_registration






