FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y \
  vim \
  git \
  make

WORKDIR /AlzheimersClassification

COPY . .

RUN pip install -r requirements.txt