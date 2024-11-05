# Profiler for training detection models 

## Introduction

This is a tool to profile the execution of training detection model on a given hardware platform, preferably on a GPUs. 
It uses PyTorch and the simple profiler employing this framework.

## Install NVidia runtime for docker

It is necessary to have an nvidia runtime. To install this runtime,
run the following commands:

```bash
sudo apt-get install nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For additional information, we refer to the following 
[link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Tool configuration 

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

To build the docker image, run the following command (takes 10 minutes to finish this build):

```bash
docker build -t dnn-profiler .
```

To start the docker container having access to the first GPU, run the following command:

```bash
docker run --rm -it --shm-size=150g --gpus '"device=0"' --name profiler-gpu-0 -v ${PWD}/.config:/app/.config dnn-profiler
```

After the container is started, it automatically runs the profiler. The output of the profiler is shown below: 

```bash
{'num_epochs': 1, 'batch_size': 2, 'prefetch_factor': 2, 'num_workers': 4}
Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
100.0%
Downloading https://thor.robots.ox.ac.uk/pets/images.tar.gz to data/oxford-iiit-pet/oxford-iiit-pet/images.tar.gz
100.0%
Extracting data/oxford-iiit-pet/oxford-iiit-pet/images.tar.gz to data/oxford-iiit-pet/oxford-iiit-pet
Downloading https://thor.robots.ox.ac.uk/pets/annotations.tar.gz to data/oxford-iiit-pet/oxford-iiit-pet/annotations.tar.gz
100.0%
Extracting data/oxford-iiit-pet/oxford-iiit-pet/annotations.tar.gz to data/oxford-iiit-pet/oxford-iiit-pet
Epoch 1/1
|█                                       | ▆█▆ 47/1840 [3%] in 29s (~18:10, 1.6/s)
```
