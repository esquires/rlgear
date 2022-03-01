FROM ubuntu:20.04

MAINTAINER Eric Squires

ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y python3-venv python3-dev git build-essential g++ python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# create venv
RUN python3 -m venv ~/venvs/rlgear
RUN source ~/venvs/rlgear/bin/activate && \
  pip install -U wheel pip setuptools && \
  rm -rf ~/.cache/pip

# install pytorch independently so
# as not to mess with neural network libraries.
# install the latest numpy because of this (https://stackoverflow.com/a/18204349).
# also install grpcio here so local builds don't take as long
RUN source ~/venvs/rlgear/bin/activate && \
  pip install -U torch torchvision numpy grpcio && \
  rm -rf ~/.cache/pip

# install dependencies in a separate step
COPY ./setup.py /root/rlgear_deps/setup.py
RUN mkdir /root/rlgear_deps/rlgear
RUN source ~/venvs/rlgear/bin/activate && pip install /root/rlgear_deps

RUN source ~/venvs/rlgear/bin/activate && python --version
COPY ./ /root/rlgear
WORKDIR /root/rlgear

RUN source ~/venvs/rlgear/bin/activate && \
  pip install pygame && \
  pip install . && \
  rm -rf ~/.cache/pip

RUN source ~/venvs/rlgear/bin/activate && \
  pip install -U .[test] 'pytest-xdist[psutil]' pytest-timeout && \
  pip install . && \
  rm -rf ~/.cache/pip

RUN source ~/venvs/rlgear/bin/activate && \
  pytest --timeout 300 -n auto -v tests && \
  rm -rf ~/ray_results
