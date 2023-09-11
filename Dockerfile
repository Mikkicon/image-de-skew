FROM python:3.10 as deskew

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR app

RUN python -m venv venv

RUN source venv/bin/activate

COPY requirements.txt ./

RUN python -m pip install -r requirements.txt

COPY invoices_rotated ./

COPY .env ./

COPY src src

COPY README.md ./

COPY generate_dataset.sh ./

RUN ./generate_dataset.sh

CMD [ "tail", "-f", "/dev/null" ]
