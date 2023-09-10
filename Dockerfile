FROM python:3.10 as deskew
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR app

RUN python -m venv venv
RUN source venv/bin/activate
# RUN python -m pip install -r requirements.txt