# python image
FROM python:3.10.6

# create working directory
RUN mkdir -p /cfos_app

COPY requirements.txt /cfos_app/requirements.txt
COPY setup.py /cfos_app/setup.py
COPY README.md /cfos_app/README.md
WORKDIR /cfos_app/

# install python packages
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN pip install -r requirements.txt
RUN pip install .

ENV PYTHONPATH "${PYTHONPATH}:/cfos_app"


# copy dictories and files 
COPY downsample/ /cfos_app/downsample/
COPY inference/ /cfos_app/inference/
COPY models/ /cfos_app/models/
COPY *.py . 
COPY *.json . 






