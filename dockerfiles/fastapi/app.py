import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import List, Optional
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from src.perfume_features import row_from_request, transform

MODEL_NAME = "perfume_accord_model_prod"
MODEL_ALIAS = "champion"


def load_model(model_name: str, alias: str):
    """
    Carga el modelo entrenado y el diccionario de features asociado.

    Intenta cargar el modelo con el alias dado desde el registro de MLflow. Si el registro
    todavía no tiene un modelo con ese alias (por ejemplo, la primera vez que se levanta el
    stack, antes de correr `notebook_example/experiment_mlflow.ipynb`), cae a un modelo por
    defecto empaquetado en la imagen (`/app/files/model.pkl`). Lo mismo para el diccionario
    de features (`data_info/data.json` en S3, con fallback a `/app/files/data.json`).
    """

    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except Exception:
        # Todavía no hay modelo en el registro de MLflow: usamos el default empaquetado.
        with open("/app/files/model.pkl", "rb") as file_ml:
            model_ml = pickle.load(file_ml)
        version_model_ml = 0

    try:
        s3 = boto3.client("s3")
        s3.head_object(Bucket="data", Key="data_info/data.json")
        result_s3 = s3.get_object(Bucket="data", Key="data_info/data.json")
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)
    except Exception:
        with open("/app/files/data.json", "r") as file_s3:
            data_dictionary = json.load(file_s3)

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Revisa si cambió la versión del modelo campeón en el registro y, si es así, recarga el
    modelo y el diccionario de features. Se llama como tarea de background después de cada
    predicción, así el DAG `retrain_the_model` puede promover un nuevo campeón sin que haga
    falta reiniciar el contenedor de la API.
    """

    global model
    global data_dict
    global version_model

    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        new_model_data = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        new_version_model = int(new_model_data.version)

        if new_version_model != version_model:
            model, version_model, data_dict = load_model(MODEL_NAME, MODEL_ALIAS)
    except Exception:
        pass


class ModelInput(BaseModel):
    """
    Input del clasificador de familia olfativa.

    Las tres listas de notas son la composición del perfume por capa de la pirámide olfativa
    (ver TP de Aprendizaje de Máquina I, sección 1.1). `release_year` y `concentration` son
    metadata opcional; si no se mandan, se imputan con la mediana/categoría "Unknown" que se
    calculó en entrenamiento, igual que se hace con los faltantes del dataset de Parfumo.

    :param top_notes: Notas de salida (ej. ["Bergamot", "Lemon"]).
    :param middle_notes: Notas de corazón (ej. ["Jasmine", "Rose"]).
    :param base_notes: Notas de fondo (ej. ["Musk", "Sandalwood", "Vanilla"]).
    :param release_year: Año de lanzamiento (opcional).
    :param concentration: Concentración (ej. "Eau de Parfum", "Eau de Toilette"; opcional).
    """

    top_notes: List[str] = Field(
        default_factory=list,
        description="Notas de salida del perfume",
    )
    middle_notes: List[str] = Field(
        default_factory=list,
        description="Notas de corazón del perfume",
    )
    base_notes: List[str] = Field(
        default_factory=list,
        description="Notas de fondo del perfume",
    )
    release_year: Optional[int] = Field(
        default=None,
        description="Año de lanzamiento del perfume (opcional)",
        ge=1850,
        le=2100,
    )
    concentration: Optional[str] = Field(
        default=None,
        description="Concentración del perfume, ej. 'Eau de Parfum' (opcional)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "top_notes": ["Bergamot", "Lemon"],
                    "middle_notes": ["Jasmine", "Rose"],
                    "base_notes": ["Musk", "Sandalwood", "Vanilla"],
                    "release_year": 2026,
                    "concentration": "Eau de Parfum",
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output del clasificador de familia olfativa.

    :param dominant_accord: Familia olfativa predicha (ej. "Floral", "Woody").
    :param top_probabilities: Hasta 3 familias con mayor probabilidad según el modelo, si el
        modelo elegido soporta `predict_proba` (los modelos basados en árboles/ensambles sí;
        SVC sin `probability=True` no). Vacío en ese caso.
    """

    dominant_accord: str = Field(
        description="Familia olfativa dominante predicha",
    )
    top_probabilities: dict = Field(
        default_factory=dict,
        description="Hasta 3 familias con mayor probabilidad y su score (si el modelo la expone)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dominant_accord": "Floral",
                    "top_probabilities": {"Floral": 0.42, "Sweet": 0.18, "Fruity": 0.09},
                }
            ]
        }
    }


# Cargamos el modelo antes de levantar la API.
model, version_model, data_dict = load_model(MODEL_NAME, MODEL_ALIAS)

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Endpoint raíz de la API. Confirma que el servicio está corriendo.
    """
    return JSONResponse(
        content=jsonable_encoder({"message": "Welcome to the Perfume Olfactory-Family Classifier API"})
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks,
):
    """
    Predice la familia olfativa dominante de un perfume a partir de su composición de notas
    (top/middle/base) y metadata opcional (año, concentración).

    Transforma el request con `src/perfume_features.py`, exactamente la misma lógica que se
    usó para construir las features de entrenamiento (ver `notebook_example/experiment_mlflow.ipynb`
    y `airflow/dags/etl_process.py`), para evitar el training-serving skew.
    """

    row = row_from_request(
        top_notes=features.top_notes,
        middle_notes=features.middle_notes,
        base_notes=features.base_notes,
        release_year=features.release_year,
        concentration=features.concentration,
    )

    X_new, _, _ = transform(row, data_dict)

    prediction = model.predict(X_new)
    dominant_accord = str(prediction[0])

    top_probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        classes = model.classes_
        top_idx = np.argsort(proba)[::-1][:3]
        top_probabilities = {str(classes[i]): round(float(proba[i]), 4) for i in top_idx}

    # Revisa en background si cambió el campeón en el registro (no bloquea esta respuesta).
    background_tasks.add_task(check_model)

    return ModelOutput(dominant_accord=dominant_accord, top_probabilities=top_probabilities)
