FROM python:3.12-slim

WORKDIR /app

# El build context es la raíz del repo (ver docker-compose.yaml), no esta carpeta, para poder
# copiar también src/perfume_features.py: es el mismo módulo de feature engineering que usa el
# notebook de entrenamiento, así el request de la API se transforma exactamente igual que en
# entrenamiento.
COPY dockerfiles/fastapi/requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

COPY dockerfiles/fastapi/ .
COPY src/ ./src/

EXPOSE 8800
