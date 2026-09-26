FROM python:3.12-slim

WORKDIR /home/jovyan/notebook_example

COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8888
