# Profiler for training detection models 

## Introduction

This is a tool to profile the execution of training detection model on a given hardware platform, preferably on a GPUs. 
It uses PyTorch and the simple profiler employing this framework.

## Install NVidia runtime for docker

To install nvidia runtime for docker, run the following command:

```bash
sudo apt-get install nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For additional information, please refer to the following [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Configuration of the profiler

The configuration file is located at `.config/config.toml`. The configuration file contains the following parameters:

- `num_epochs`: The number of epochs - the number of times the entire dataset is passed through the model.
- `batch_size`: The batch size - the number of samples in a batch.
- `prefetch_factor`: The prefetch factor - the number of batches to be prefetched.
- `num_workers`: The number of workers - the number of processes to be used for data loading.
- `device`: The device to be used for training. It can be either `cpu` or `cuda:<device_id>`, default is `cuda:0`.


The structure of the configuration file is as follows:

```toml
[training]
num_epochs = 1
batch_size = 2
prefetch_factor = 2
num_workers = 4
device = "cuda:0"
```


## Build docker image and run the container

To build the docker image, run the following command:

```bash
docker build -t deep-learning-profiler .
```

To run the docker container for GPU having id 0, run the following command:

```bash
docker run --gpus '"device=0"' -it deep-learning-profiler --name profiler-gpu-0 -d
```
