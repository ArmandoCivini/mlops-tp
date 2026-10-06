"""Contrato del ETL, al estilo *golden dataset*.

Estos tests no miden la calidad del modelo: verifican que la limpieza del catálogo de
Parfumo siga respetando las reglas que se definieron en la Sección 2 del TP de Aprendizaje
de Máquina I. Si alguien cambia `src/perfume_features.py` y, sin darse cuenta, deja de
deduplicar o deja entrar el artefacto de scraping, acá se ve.
"""

import pandas as pd

from conftest import SAMPLE_MIN_SAMPLES_PER_CLASS

COLUMNAS_OBLIGATORIAS = ["Top_Notes", "Middle_Notes", "Base_Notes", "Main_Accords"]


def test_descarta_el_artefacto_de_scraping(gold_sample: pd.DataFrame):
    """"Main Accords" no es una familia olfativa: es el encabezado de la tabla que quedó
    pegado al scrapear Parfumo."""
    assert "Main Accords" not in gold_sample["dominant_accord"].unique()


def test_no_quedan_perfumes_duplicados(gold_sample: pd.DataFrame):
    """La deduplicación es por (Name, Brand): el mismo perfume aparece varias veces en el
    catálogo, y si no se filtra, el mismo registro cae en train y en test a la vez."""
    duplicados = gold_sample.duplicated(subset=["Name", "Brand"]).sum()

    assert duplicados == 0


def test_todas_las_clases_superan_el_minimo_de_muestras(gold_sample: pd.DataFrame):
    """Las clases muy chicas se descartan: con menos muestras que el umbral, el f1 de esa
    familia es ruido."""
    conteos = gold_sample["dominant_accord"].value_counts()

    assert conteos.min() >= SAMPLE_MIN_SAMPLES_PER_CLASS


def test_no_hay_filas_sin_notas_ni_accords(gold_sample: pd.DataFrame):
    """El modelo predice a partir de la pirámide olfativa completa, así que una fila sin una
    de las tres capas no sirve."""
    assert gold_sample[COLUMNAS_OBLIGATORIAS].isna().sum().sum() == 0


def test_el_target_nunca_es_nulo(gold_sample: pd.DataFrame):
    assert gold_sample["dominant_accord"].notna().all()


def test_los_anios_invalidos_quedan_como_nan(gold_sample: pd.DataFrame):
    """Parfumo tiene años imposibles (un perfume de 1800 y pico es dato sucio, no histórico).
    Se convierten a NaN para que después se imputen con la mediana."""
    anios = gold_sample["Release_Year"].dropna()

    assert (anios >= 1850).all()


def test_las_cuentas_de_notas_son_coherentes(gold_sample: pd.DataFrame):
    """`n_notas_total` es una feature derivada: tiene que ser la suma de las tres capas."""
    suma = gold_sample["n_top"] + gold_sample["n_mid"] + gold_sample["n_base"]

    assert (gold_sample["n_notas_total"] == suma).all()
    assert (gold_sample["n_top"] > 0).all()


def test_bajar_el_minimo_de_muestras_nunca_deja_menos_clases(raw_sample: pd.DataFrame):
    """El umbral tiene que comportarse de forma monótona: con un mínimo más permisivo no
    puede haber *menos* clases. Es el chequeo de que el parámetro realmente se usa y no
    quedó una constante hardcodeada adentro del módulo."""
    from perfume_features import build_gold_dataset

    permisivo = build_gold_dataset(raw_sample, min_samples_per_class=5)
    estricto = build_gold_dataset(raw_sample, min_samples_per_class=50)

    assert permisivo["dominant_accord"].nunique() >= estricto["dominant_accord"].nunique()
    assert len(permisivo) >= len(estricto)


def test_el_parametro_de_k_notas_cambia_el_tamanio_del_vocabulario(gold_sample: pd.DataFrame):
    """Igual que el anterior, pero para `k_notes_per_layer`: el DAG lo lee de una Airflow
    Variable, así que tiene que tener efecto de verdad."""
    from perfume_features import fit_feature_spec

    spec_chico = fit_feature_spec(gold_sample, k_notes_per_layer=10)
    spec_grande = fit_feature_spec(gold_sample, k_notes_per_layer=80)

    assert len(spec_chico["vocab_top"]) == 10
    assert len(spec_grande["vocab_top"]) > len(spec_chico["vocab_top"])
    assert len(spec_chico["feature_names"]) < len(spec_grande["feature_names"])
