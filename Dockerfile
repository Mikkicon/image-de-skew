FROM python:3.10 as deskew

SHELL ["/bin/bash", "-c"]

# install python open CV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# move to app 
WORKDIR app

# create a virtual environment for python to isolate the dependencies 
RUN python -m venv venv
# enter the created virtual environment 
RUN source venv/bin/activate

# copy required libraries to container 
COPY requirements.txt ./
# install the required libraries 
RUN python -m pip install -r requirements.txt

# copy files 
COPY invoices_rotated ./
COPY .env ./
COPY src src
COPY README.md ./
COPY generate_dataset.sh ./

# run the generate_dataset.sh script
#       which fetches zip archive with document images from internet
#       and then skews them by some random degree 
RUN ./generate_dataset.sh

# keep the container running 
CMD [ "tail", "-f", "/dev/null" ]
