"""Fixtures compartidas por los tests.

`src/` no es un paquete instalado, así que se agrega al sys.path acá (misma idea que
hace la tarea `build_features` del DAG de ETL con /opt/airflow/repo_src).
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_CSV = REPO_ROOT / "data" / "parfumo_perfumes.csv"
API_DATA_JSON = REPO_ROOT / "dockerfiles" / "fastapi" / "files" / "data.json"
API_MODEL_PKL = REPO_ROOT / "dockerfiles" / "fastapi" / "files" / "model.pkl"

# Para los tests se lee solo una porción del catálogo (el CSV completo son 59.325 filas) y
# se baja el mínimo de muestras por clase en consecuencia: con 4.000 filas, el umbral de 100
# del TP dejaría el dataset casi vacío y los tests no probarían nada.
SAMPLE_ROWS = 4000
SAMPLE_MIN_SAMPLES_PER_CLASS = 20


@pytest.fixture(scope="session")
def api_spec() -> dict:
    """El `data.json` que viaja en la imagen de FastAPI: el mismo diccionario de features
    que la API usa para transformar un request."""
    with open(API_DATA_JSON, encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """Muestra del catálogo crudo de Parfumo."""
    return pd.read_csv(RAW_CSV, nrows=SAMPLE_ROWS)


@pytest.fixture(scope="session")
def gold_sample(raw_sample: pd.DataFrame) -> pd.DataFrame:
    """La muestra ya limpia, con el target y las columnas derivadas."""
    from perfume_features import build_gold_dataset

    return build_gold_dataset(
        raw_sample,
        min_samples_per_class=SAMPLE_MIN_SAMPLES_PER_CLASS,
    )


@pytest.fixture(scope="session")
def sample_spec(gold_sample: pd.DataFrame) -> dict:
    """Spec de features fiteado sobre la muestra limpia."""
    from perfume_features import fit_feature_spec

    return fit_feature_spec(gold_sample)


@pytest.fixture(scope="session")
def packaged_model():
    """El modelo de fallback empaquetado en la imagen de FastAPI.

    Se saltea el test si el pickle no se puede deserializar con las versiones instaladas:
    el modelo se entrenó con numpy 2.x y scikit-learn 1.8, así que con numpy 1.x la carga
    falla con `PCG64 is not a known BitGenerator module`. No es un bug del modelo, es el
    entorno de desarrollo que no replica al productivo (ver README, sección de tests).
    """
    try:
        with open(API_MODEL_PKL, "rb") as file:
            return pickle.load(file)
    except Exception as error:  # noqa: BLE001 - cualquier fallo de deserialización
        pytest.skip(
            "No se pudo deserializar dockerfiles/fastapi/files/model.pkl con las versiones "
            f"instaladas ({type(error).__name__}: {error}). Hace falta numpy>=2 y, idealmente, "
            "scikit-learn 1.8 como en los contenedores."
        )
