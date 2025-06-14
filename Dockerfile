FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -e .

## GPU ka budget nahi hai, will test after CPU

# FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
# RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip build-essential libgomp1 \
#     && ln -s /usr/bin/python3.8 /usr/bin/python \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

RUN python src/training_pipeline.py

EXPOSE 5000

CMD ["python3", "application.py"]