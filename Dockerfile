FROM ubuntu:18.04

MAINTAINER Eric Squires

ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y python3-pip python3-pytest git python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# install tensorflow and pytorch independently so
# as not to mess with neural network libraries
RUN pip3 install tensorflow torch torchvision mypy flake8 pylint pydocstyle

COPY ./ /root/rlgear
WORKDIR /root/rlgear
RUN rm -rf $(find -name '*.pyc' -o -name '__pycache__')
RUN pip3 install .
RUN py.test-3 tests
