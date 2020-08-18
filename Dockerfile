FROM ubuntu:18.04

MAINTAINER Eric Squires

ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y python3-pip python3-pytest git python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# install pytorch independently so
# as not to mess with neural network libraries.
# install the latest numpy because of this (https://stackoverflow.com/a/18204349).
# also install grpcio here so local builds don't take as long
RUN pip3 install -U torch torchvision numpy grpcio

COPY ./ /root/rlgear
WORKDIR /root/rlgear
RUN rm -rf $(find -name '*.pyc' -o -name '__pycache__')
RUN pip3 install .

RUN pip3 install -U mypy flake8 pylint pydocstyle
RUN py.test-3 tests
