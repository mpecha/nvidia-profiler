FROM python:3.11-slim

# copy app source
RUN mkdir -p /app/.config
WORKDIR /app/nvidia_profiler

# copy requirements
COPY requirements.txt /app/nvidia_profiler/requirements.txt

# copy app source
COPY nvidia_profiler/config.py /app/nvidia_profiler/
COPY nvidia_profiler/data.py /app/nvidia_profiler/
COPY nvidia_profiler/pipeline.py /app/nvidia_profiler/

# install ffmpeg
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/nvidia_profiler/requirements.txt

# set add path of application for python interpreter
ENV PYTHONPATH=$PYTHONPATH:/app

# run the profiler
ENTRYPOINT ["sh", "-c", "python pipeline.py --config /app/.config/train.toml"]