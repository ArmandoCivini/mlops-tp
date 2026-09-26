import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Perfume Olfactory-Family Data

Este DAG toma el catálogo crudo de Parfumo (59.325 perfumes, `data/parfumo_perfumes.csv`, montado
como volumen de solo lectura en el contenedor) y arma el dataset de "familia olfativa dominante"
tal como se definió en el TP de Aprendizaje de Máquina I *"Clasificación de familia olfativa en
perfumes a partir de su composición de notas"* (Francisco Facchini, nota 10/10): se descartan
perfumes sin notas o accords completos, se deduplica por (Name, Brand), se descartan clases con
menos de 100 muestras, y se construyen las features (240 binarias de notas top/middle/base + 6
numéricas + dummies de concentración).

El target es `dominant_accord`: la primera familia olfativa de la lista `Main_Accords` que declara
Parfumo para cada perfume (la convención de la plataforma es ordenarlas de mayor a menor
intensidad percibida).

Guarda el resultado (train/test + diccionario de features para que `dockerfiles/fastapi/app.py`
pueda transformar requests nuevos de la misma manera) en el bucket `s3://data`, y trackea el
dataset en MLflow, experimento `Perfume Olfactory Family`.

Ver `src/perfume_features.py` en la raíz del repo para la versión documentada y reutilizable de
esta misma lógica (se reimplementa acá en línea porque las tareas `@task.virtualenv` corren en un
venv aislado por tarea y no pueden importar módulos locales del repo).
"""

default_args = {
    "owner": "Francisco Facchini",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=20),
}


@dag(
    dag_id="process_perfume_data",
    description="ETL process for the perfume olfactory-family dataset (Parfumo), separating into train/test.",
    doc_md=markdown_text,
    tags=["ETL", "Perfume Astral"],
    default_args=default_args,
    catchup=False,
)
def process_perfume_data():

    @task.virtualenv(
        task_id="get_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def get_data():
        """
        Lee el CSV crudo de Parfumo desde el volumen local montado en el contenedor
        (./data/parfumo_perfumes.csv -> /opt/airflow/data_raw/parfumo_perfumes.csv, ver
        docker-compose.yaml) y lo sube al bucket de datos `s3://data`.

        A diferencia del ejemplo de Heart Disease (que descarga en vivo desde el repositorio
        de UCI), acá no hay una API pública de la que scrapear Parfumo en cada corrida del
        DAG, así que el crudo se versiona como archivo local y esta tarea solo lo publica en
        el Data Lake simulado.
        """
        import pandas as pd
        import awswrangler as wr

        local_path = "/opt/airflow/data_raw/parfumo_perfumes.csv"
        data_path = "s3://data/raw/parfumo_perfumes.csv"

        dataframe = pd.read_csv(local_path)
        wr.s3.to_csv(df=dataframe, path=data_path, index=False)

    @task.virtualenv(
        task_id="build_features",
        requirements=["awswrangler==3.6.0", "mlflow==3.1.4"],
        system_site_packages=True,
    )
    def build_features():
        """
        Limpieza + feature engineering (Sección 2 del TP de AMq1): dropea filas sin notas o
        accords completos, descarta el artefacto de scraping, deduplica, descarta clases
        chicas, arma el target y las 256 features (240 notas multi-hot + 6 numéricas +
        dummies de concentración). Guarda la matriz final y el diccionario de features
        (`data_info/data.json`) que usan tanto `split_dataset` como la API para transformar
        requests nuevos de la misma manera.
        """
        import json
        import datetime as dt
        from collections import Counter

        import boto3
        import mlflow
        import numpy as np
        import pandas as pd
        import awswrangler as wr
        from airflow.models import Variable

        min_samples = int(Variable.get("min_samples_per_class_perfume"))
        k_notes = int(Variable.get("k_notes_per_layer_perfume"))
        n_top_conc = int(Variable.get("n_top_concentrations_perfume"))
        min_year = int(Variable.get("min_valid_year_perfume"))

        def split_notes(s, sep=","):
            if pd.isna(s):
                return []
            return [x.strip() for x in str(s).split(sep) if x.strip() and x.strip() != "NA"]

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

        df = wr.s3.read_csv("s3://data/raw/parfumo_perfumes.csv")
        df["Release_Year"] = pd.to_numeric(df["Release_Year"], errors="coerce")

        gold = df.dropna(subset=["Top_Notes", "Middle_Notes", "Base_Notes", "Main_Accords"]).copy()
        gold["accords_list"] = gold["Main_Accords"].apply(split_notes)
        gold["dominant_accord"] = gold["accords_list"].apply(lambda l: l[0] if l else None)
        gold = gold[gold["dominant_accord"] != "Main Accords"]  # artefacto de scraping
        gold = gold.drop_duplicates(subset=["Name", "Brand"])

        vc = gold["dominant_accord"].value_counts()
        keep_classes = vc[vc >= min_samples].index.tolist()
        gold = gold[gold["dominant_accord"].isin(keep_classes)].copy()

        for col, name in [("Top_Notes", "top"), ("Middle_Notes", "mid"), ("Base_Notes", "base")]:
            gold[f"{name}_list"] = gold[col].apply(split_notes)
            gold[f"n_{name}"] = gold[f"{name}_list"].apply(len)
        gold["n_notas_total"] = gold["n_top"] + gold["n_mid"] + gold["n_base"]
        gold["n_accords"] = gold["accords_list"].apply(len)
        gold.loc[gold["Release_Year"] < min_year, "Release_Year"] = np.nan

        vocab_top = top_k_vocab(gold["top_list"], k_notes)
        vocab_mid = top_k_vocab(gold["mid_list"], k_notes)
        vocab_base = top_k_vocab(gold["base_list"], k_notes)

        num_cols = ["Release_Year", "n_top", "n_mid", "n_base", "n_notas_total", "n_accords"]
        year_median = float(gold["Release_Year"].median())
        n_accords_median = float(gold["n_accords"].median())

        top_conc = gold["Concentration"].value_counts().head(n_top_conc).index.tolist()
        conc_categories = top_conc + ["Unknown", "Otras"]
        conc = gold["Concentration"].fillna("Unknown")
        conc = conc.where(conc.isin(top_conc + ["Unknown"]), "Otras")
        X_conc = pd.get_dummies(pd.Categorical(conc, categories=conc_categories), prefix="conc")
        for c in conc_categories:
            col = f"conc_{c}"
            if col not in X_conc.columns:
                X_conc[col] = 0
        X_conc = X_conc[[f"conc_{c}" for c in conc_categories]]

        X_notes = np.hstack([
            multi_hot(gold["top_list"], vocab_top),
            multi_hot(gold["mid_list"], vocab_mid),
            multi_hot(gold["base_list"], vocab_base),
        ])
        X_num = gold[num_cols].copy()
        X_num["Release_Year"] = X_num["Release_Year"].fillna(year_median)
        X_num["n_accords"] = X_num["n_accords"].fillna(n_accords_median)

        feature_names = (
            [f"top__{w}" for w in vocab_top]
            + [f"mid__{w}" for w in vocab_mid]
            + [f"base__{w}" for w in vocab_base]
            + num_cols
            + list(X_conc.columns)
        )

        X = np.hstack([X_notes.astype(float), X_num.to_numpy(dtype=float), X_conc.to_numpy(dtype=float)])
        features_df = pd.DataFrame(X, columns=feature_names)
        features_df["dominant_accord"] = gold["dominant_accord"].to_numpy()

        wr.s3.to_csv(df=features_df, path="s3://data/final/parfumo_features.csv", index=False)

        data_dict = {
            "k_notes_per_layer": k_notes,
            "vocab_top": vocab_top,
            "vocab_mid": vocab_mid,
            "vocab_base": vocab_base,
            "num_cols": num_cols,
            "year_median": year_median,
            "n_accords_median": n_accords_median,
            "top_concentrations": top_conc,
            "conc_categories": conc_categories,
            "feature_names": feature_names,
            "classes": sorted(gold["dominant_accord"].unique().tolist()),
            "n_rows": int(features_df.shape[0]),
            "n_features": int(len(feature_names)),
            "date": dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"),
        }
        data_string = json.dumps(data_dict, indent=2)

        s3 = boto3.client("s3", endpoint_url="http://s3:9000")
        s3.put_object(Bucket="data", Key="data_info/data.json", Body=data_string)

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("Perfume Olfactory Family")

        with mlflow.start_run(
            run_name="ETL_run_" + dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Parfumo perfumes"},
            log_system_metrics=True,
        ):
            mlflow.log_param("n_rows_gold", features_df.shape[0])
            mlflow.log_param("n_classes", gold["dominant_accord"].nunique())
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("min_samples_per_class", min_samples)
            mlflow.log_param("k_notes_per_layer", k_notes)

            mlflow_dataset = mlflow.data.from_pandas(
                features_df,
                source="https://www.parfumo.com/",
                targets="dominant_accord",
                name="parfumo_features",
            )
            mlflow.log_input(mlflow_dataset, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def split_dataset():
        """
        Split estratificado 80/20 sobre la matriz de features ya armada, guardando
        X_train/y_train/X_test/y_test en `s3://data/final/{train,test}`.
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        test_size = float(Variable.get("test_size_perfume"))
        target_col = Variable.get("target_col_perfume")

        dataset = wr.s3.read_csv("s3://data/final/parfumo_features.csv")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        save_to_csv(X_train, "s3://data/final/train/perfume_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/perfume_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/perfume_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/perfume_y_test.csv")

    get_data() >> build_features() >> split_dataset()


dag = process_perfume_data()
