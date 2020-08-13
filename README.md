# RL Gear

[![Build Status](https://travis-ci.com/esquires/rlgear.svg?branch=master)](https://travis-ci.com/esquires/rlgear)

This project makes setting up new research projects with
[ray](https://docs.ray.io/en/latest/index.html) a bit more turn-key.

## Installation

`rlgear` is designed to work with ray version 0.8.7 or later
and has been tested with python 3.6.

First install `tensorflow` and `pytorch`. cpu-only `tensorflow` is
fine since `rlgear` doesn't use any neural network
operations from `tensorflow`.

Second, you may need to install opencv since `ray/rllib/env/atari_wrappers.py`
requires the `cv2` package but `opencv` is not installed automatically with
`ray`. It can be installed on Ubuntu with `apt install python3-opencv`.

Thid, install the package.
```bash
    pip install .
```

See the `Dockerfile` for a minimal example of how to install on Ubuntu 18.04.
To build it locally, run
```bash
    docker build -t rlgear:latest .
```

## Usage

See `tests/test_train_cartpole.py` for a minimal working example.

## Features

### Canonical networks

Common networks such as DQN and IMPALA are implemented in pytorch
as well as a fully connected network that has separate networks
for the value and policy. There is also a helper class to reduce
boilerplate code for feedforward networks. See `torch_models.py`.

### Setting Up Experiments

Import yaml files from other yaml files to adjust a small portion
for a new experiment or save meta data from an experiment (git info,
requirements.txt, etc). See `utils.py` and `rllib_utils.py`)

### Tensorboard Plotting

After running an experiment multiple times, plot it in matplotlib
with transparent percentiles. See `scripts.py` and `utils.py`


## License

BSD-3-Clause
