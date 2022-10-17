# python image
FROM python:3.10.6

# create working directory
RUN mkdir -p /cfos_app

COPY requirements.txt /cfos_app/requirements.txt
WORKDIR /cfos_app/

# install python packages
RUN pip install -r requirements.txt


# copy dictories and files 
COPY downsample/ /cfos_app/downsample/
COPY inference/ /cfos_app/inference/
COPY models/ /cfos_app/models/
COPY *.py . 






