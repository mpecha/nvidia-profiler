FROM python:3.11-slim

# copy config file
WORKDIR /app/.config/
COPY .config/train.toml /app/.config/train.toml
COPY requirements.txt /app/.config/requirements.txt

# copy app source
WORKDIR /app/nvidia_profiler
COPY nvidia_profiler/config.py /app/nvidia_profiler/
COPY nvidia_profiler/data.py /app/nvidia_profiler/
COPY nvidia_profiler/pipeline.py /app/nvidia_profiler/

# install ffmpeg
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/.config/requirements.txt

# set add path of application for python interpreter
ENV PYTHONPATH=$PYTHONPATH:/app

# set working directory
WORKDIR /app/nvidia_profiler
ENTRYPOINT ["sh", "-c", "python pipeline.py --config /app/.config/train.toml"]