"""Golden numbers del dataset completo.

Verifica que el catálogo entero de Parfumo siga produciendo exactamente los números que
reporta el README y que validamos contra el TP de AMq1: 21.032 perfumes limpios, 256
features y 20 clases. Es la red de seguridad más fuerte que tenemos contra un cambio
silencioso en la limpieza o en el feature engineering.

Procesa las 59.325 filas del catálogo, pero las transformaciones están vectorizadas: tarda
alrededor de un segundo, así que corre junto con el resto de la suite.
"""

import pandas as pd
import pytest

from conftest import RAW_CSV
from perfume_features import build_gold_dataset, fit_feature_spec, transform

# Los valores del TP de AMq1, tal como los reporta el README.
FILAS_ESPERADAS = 21032
FEATURES_ESPERADAS = 256
CLASES_ESPERADAS = 20


@pytest.fixture(scope="module")
def dataset_completo():
    """El pipeline completo con los parámetros reales del TP (los defaults del módulo, que
    son los mismos valores que el DAG lee de las Airflow Variables)."""
    crudo = pd.read_csv(RAW_CSV)
    gold = build_gold_dataset(crudo)
    spec = fit_feature_spec(gold)
    X, y, feature_names = transform(gold, spec)

    return gold, spec, X, y, feature_names


def test_el_dataset_limpio_tiene_las_filas_del_readme(dataset_completo):
    gold, _, _, _, _ = dataset_completo

    assert len(gold) == FILAS_ESPERADAS


def test_la_matriz_tiene_las_features_del_readme(dataset_completo):
    _, spec, X, _, feature_names = dataset_completo

    assert X.shape == (FILAS_ESPERADAS, FEATURES_ESPERADAS)
    assert len(feature_names) == FEATURES_ESPERADAS
    assert spec["feature_names"] == feature_names


def test_quedan_las_clases_del_readme(dataset_completo):
    gold, spec, _, y, _ = dataset_completo

    assert gold["dominant_accord"].nunique() == CLASES_ESPERADAS
    assert len(spec["classes"]) == CLASES_ESPERADAS
    assert set(y) == set(spec["classes"])


def test_el_vocabulario_de_notas_esta_completo(dataset_completo):
    """240 de las 256 features son notas: 80 por capa. Si una capa tiene menos vocabulario
    que K, es que el catálogo cambió."""
    _, spec, _, _, _ = dataset_completo

    assert len(spec["vocab_top"]) == spec["k_notes_per_layer"]
    assert len(spec["vocab_mid"]) == spec["k_notes_per_layer"]
    assert len(spec["vocab_base"]) == spec["k_notes_per_layer"]
