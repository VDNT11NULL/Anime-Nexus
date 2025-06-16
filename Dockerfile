FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    build-essential \
    libgomp1 \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python pipelines/training_pipeline.py

EXPOSE 5000

CMD ["python3", "application.py"]