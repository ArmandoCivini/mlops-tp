"""
Lógica de limpieza y feature engineering para el clasificador de familia olfativa
(Perfume Astral / MLOps TP).

Este módulo es la única implementación de las transformaciones: no hay copias. Lo usan
los tres lados del pipeline, así que entrenamiento, ETL e inferencia no pueden
desincronizarse.
  - airflow/dags/etl_process.py (tarea `build_features`). Las tareas decoradas con
    @task.virtualenv corren en un venv aislado, que no tiene el repo en el sys.path: el
    directorio se monta en el contenedor como /opt/airflow/repo_src (ver `volumes` de
    x-airflow-common en docker-compose.yaml) y la tarea lo agrega al sys.path antes de
    importar este módulo.
  - notebook_example/experiment_mlflow.ipynb, para prototipar y entrenar.
  - dockerfiles/fastapi/app.py, para transformar el request de la API con el mismo criterio
    que se usó en entrenamiento (el Dockerfile copia `src/` a la imagen).

Los valores por defecto de los parámetros de abajo son los del TP de AMq1. El DAG de ETL
los pasa explícitos, leyéndolos de las Airflow Variables (`airflow/secrets/variables.yaml`),
así que se pueden cambiar sin tocar código.

Basado en la Sección 2 del TP de Aprendizaje de Máquina I ("Clasificación de familia
olfativa en perfumes a partir de su composición de notas", Francisco Facchini, nota 10/10).
"""

from collections import Counter

import numpy as np
import pandas as pd

# Cantidad de notas más frecuentes que se conservan por capa (ver 2.8 del TP de AMq1).
K_NOTES_PER_LAYER = 80
# Umbral mínimo de muestras por clase para conservarla (ver 2.4 del TP de AMq1).
MIN_SAMPLES_PER_CLASS = 100
# Categorías de concentración más frecuentes a mantener explícitas (el resto -> "Otras").
N_TOP_CONCENTRATIONS = 8
MIN_VALID_YEAR = 1850


def split_notes(s, sep=","):
    """Convierte el string crudo de una capa de notas ('Bergamot, Lemon') en una lista
    de tokens limpios. Devuelve [] si el valor es NaN o vacío."""
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(sep) if x.strip() and x.strip() != "NA"]


def build_gold_dataset(
    df: pd.DataFrame,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    min_valid_year: int = MIN_VALID_YEAR,
) -> pd.DataFrame:
    """Limpieza (Sección 2.3-2.4 del TP de AMq1): a partir del catálogo crudo de Parfumo,
    se queda con las filas que tienen notas completas en las tres capas y accords, arma el
    target `dominant_accord`, descarta el artefacto de scraping, deduplica por (Name, Brand)
    y descarta clases con menos de `min_samples_per_class` muestras.
    """
    df = df.copy()
    df["Release_Year"] = pd.to_numeric(df["Release_Year"], errors="coerce")

    gold = df.dropna(subset=["Top_Notes", "Middle_Notes", "Base_Notes", "Main_Accords"]).copy()
    gold["accords_list"] = gold["Main_Accords"].apply(split_notes)
    gold["dominant_accord"] = gold["accords_list"].apply(lambda l: l[0] if l else None)

    gold = gold[gold["dominant_accord"] != "Main Accords"]  # artefacto de scraping
    gold = gold.drop_duplicates(subset=["Name", "Brand"])  # duplicados

    vc = gold["dominant_accord"].value_counts()
    keep_classes = vc[vc >= min_samples_per_class].index.tolist()
    gold = gold[gold["dominant_accord"].isin(keep_classes)].copy()

    for col, name in [("Top_Notes", "top"), ("Middle_Notes", "mid"), ("Base_Notes", "base")]:
        gold[f"{name}_list"] = gold[col].apply(split_notes)
        gold[f"n_{name}"] = gold[f"{name}_list"].apply(len)
    gold["n_notas_total"] = gold["n_top"] + gold["n_mid"] + gold["n_base"]
    gold["n_accords"] = gold["accords_list"].apply(len)

    gold.loc[gold["Release_Year"] < min_valid_year, "Release_Year"] = np.nan

    return gold.reset_index(drop=True)


def top_k_vocab(lists, k):
    c = Counter()
    for l in lists:
        c.update(l)
    return [w for w, _ in c.most_common(k)]


def multi_hot(lists, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(lists), len(vocab)), dtype=np.int8)
    for i, l in enumerate(lists):
        for w in l:
            j = idx.get(w)
            if j is not None:
                M[i, j] = 1
    return M


def fit_feature_spec(
    gold: pd.DataFrame,
    k_notes_per_layer: int = K_NOTES_PER_LAYER,
    n_top_concentrations: int = N_TOP_CONCENTRATIONS,
) -> dict:
    """A partir del dataset limpio (train), calcula todo lo que hace falta para poder
    transformar filas nuevas de la misma manera en el futuro (vocabularios de notas,
    mediana de año, categorías de concentración top-N). Esto es el equivalente al
    `data_dict` / `data.json` del ejemplo de Heart Disease, y es exactamente lo que el DAG
    de ETL serializa a `s3://data/data_info/data.json` para que lo lea la API."""
    vocab_top = top_k_vocab(gold["top_list"], k_notes_per_layer)
    vocab_mid = top_k_vocab(gold["mid_list"], k_notes_per_layer)
    vocab_base = top_k_vocab(gold["base_list"], k_notes_per_layer)

    year_median = float(gold["Release_Year"].median())

    top_conc = gold["Concentration"].value_counts().head(n_top_concentrations).index.tolist()

    classes = sorted(gold["dominant_accord"].unique().tolist())

    num_cols = ["Release_Year", "n_top", "n_mid", "n_base", "n_notas_total", "n_accords"]

    # OJO (limitación conocida, ver README): `n_accords` (cuántas familias declara Parfumo
    # para ese perfume) se calcula a partir de `Main_Accords`, el mismo campo del que sale
    # el target `dominant_accord`. En el dataset histórico está disponible tanto en train
    # como en test porque son perfumes ya catalogados, así que no distorsiona la métrica
    # reportada en el TP. Pero para un perfume genuinamente nuevo (por ejemplo, una fórmula
    # que todavía no generó su ficha de Parfumo) ese dato no existe de antemano: es
    # justamente lo que se está tratando de predecir. Por eso en inferencia se imputa con
    # la mediana de train en vez de pedirlo como input obligatorio.
    n_accords_median = float(gold["n_accords"].median())

    spec = {
        "k_notes_per_layer": k_notes_per_layer,
        "vocab_top": vocab_top,
        "vocab_mid": vocab_mid,
        "vocab_base": vocab_base,
        "num_cols": num_cols,
        "year_median": year_median,
        "n_accords_median": n_accords_median,
        "top_concentrations": top_conc,
        # Las categorías de concentración y el orden de las columnas quedan congelados acá:
        # en inferencia se transforma una sola fila, así que no se pueden re-deducir del
        # batch (ver `transform`).
        "conc_categories": conc_categories(top_conc),
        "classes": classes,
    }
    spec["feature_names"] = feature_names_from_spec(spec)
    return spec


def conc_categories(top_concentrations):
    """Categorías de la dummy de concentración: las top-N de train, más `Unknown` (faltante)
    y `Otras` (concentración vista pero fuera del top-N)."""
    return list(top_concentrations) + ["Unknown", "Otras"]


def feature_names_from_spec(spec: dict) -> list:
    """Nombres de las columnas de X, en el mismo orden en que `transform` las apila."""
    return (
        [f"top__{w}" for w in spec["vocab_top"]]
        + [f"mid__{w}" for w in spec["vocab_mid"]]
        + [f"base__{w}" for w in spec["vocab_base"]]
        + list(spec["num_cols"])
        + [f"conc_{c}" for c in spec["conc_categories"]]
    )


def transform(gold: pd.DataFrame, spec: dict):
    """Aplica el `spec` (ya fiteado con fit_feature_spec) para construir la matriz de
    features X y el target y. Es la misma transformación tanto en train como en inferencia:
    en inferencia, `gold` es un DataFrame de una sola fila armado a partir del request de
    la API (ver dockerfiles/fastapi/app.py)."""
    X_notes = np.hstack([
        multi_hot(gold["top_list"], spec["vocab_top"]),
        multi_hot(gold["mid_list"], spec["vocab_mid"]),
        multi_hot(gold["base_list"], spec["vocab_base"]),
    ])

    X_num = gold[spec["num_cols"]].copy()
    X_num["Release_Year"] = X_num["Release_Year"].fillna(spec["year_median"])
    if "n_accords" in X_num.columns:
        X_num["n_accords"] = X_num["n_accords"].fillna(spec["n_accords_median"])

    top_conc = list(spec["top_concentrations"])
    categories = list(spec.get("conc_categories") or conc_categories(top_conc))
    conc = gold["Concentration"].fillna("Unknown")
    conc = conc.where(conc.isin(top_conc + ["Unknown"]), "Otras")
    X_conc = pd.get_dummies(pd.Categorical(conc, categories=categories), prefix="conc")
    # Aseguramos que todas las columnas de concentración existan siempre, en el mismo orden,
    # aunque en este batch no haya aparecido alguna categoría (crítico en inferencia con 1 fila).
    for c in categories:
        col = f"conc_{c}"
        if col not in X_conc.columns:
            X_conc[col] = 0
    X_conc = X_conc[[f"conc_{c}" for c in categories]]

    feature_names = spec.get("feature_names") or feature_names_from_spec(
        {**spec, "conc_categories": categories}
    )

    X = np.hstack([
        X_notes.astype(float),
        X_num.to_numpy(dtype=float),
        X_conc.to_numpy(dtype=float),
    ])

    y = gold["dominant_accord"].to_numpy() if "dominant_accord" in gold.columns else None

    return X, y, feature_names


def row_from_request(top_notes, middle_notes, base_notes, release_year=None, concentration=None):
    """Arma un DataFrame de una sola fila con la misma forma que `build_gold_dataset`
    produce por fila, a partir de los campos que manda la API. Se usa en
    dockerfiles/fastapi/app.py antes de llamar a `transform`."""
    row = {
        "top_list": list(top_notes or []),
        "mid_list": list(middle_notes or []),
        "base_list": list(base_notes or []),
        "Release_Year": release_year,
        "Concentration": concentration,
    }
    row["n_top"] = len(row["top_list"])
    row["n_mid"] = len(row["mid_list"])
    row["n_base"] = len(row["base_list"])
    row["n_notas_total"] = row["n_top"] + row["n_mid"] + row["n_base"]
    # No tenemos accords en inferencia (es lo que se predice), así que n_accords se deja en
    # NaN acá y se imputa con la mediana de train dentro de `transform` (ver nota en
    # `fit_feature_spec` sobre por qué esta feature no está disponible en producción).
    row["n_accords"] = np.nan
    return pd.DataFrame([row])
