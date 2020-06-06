FROM tensorflow/tensorflow:1.14.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

# dependencies
RUN apt-get -y update && apt-get -y install \
    build-essential \
    wget \
    tmux \
    git \
    nano \
    vim \
    libopencv-dev \
    python3-opencv

# python dependencies
RUN pip install --upgrade pip

RUN pip install Cython \
    numpy \
    matplotlib \
    seaborn \
    pandas \
    h5py \
    jupyterlab \
    ipython \
    nose \
    tqdm \
    pyyaml \
    contextlib2 \
    pillow \
    lxml \
    jupyter \
    pathlib

WORKDIR /tensorflow
RUN git clone https://github.com/ozangungortuhh/models.git

WORKDIR /tensorflow/models/research
RUN pip install .

RUN export PYTHONPATH=$PYTHONPATH:/tensorflow/models/research/slim
