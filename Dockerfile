FROM ubuntu:latest

WORKDIR /app

RUN sudo apt-get install python3
RUN sudo apt-get install default-jdk
RUN sudo apt-get install maven

COPY requirements.txt ./
RUN pip install -r requirement.txt

COPY study1 ./study1
COPY study2 ./study2
COPY run_cmnd.sh ./