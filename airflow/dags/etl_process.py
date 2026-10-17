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

La limpieza y el feature engineering viven en `src/perfume_features.py`, el mismo módulo que usan
el notebook de entrenamiento y la API: este DAG lo importa, no lo reimplementa.
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
        requirements=["awswrangler==3.6.0", "pyarrow==17.0.0"],
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
        requirements=["awswrangler==3.6.0", "pyarrow==17.0.0", "mlflow==3.1.4"],
        system_site_packages=True,
    )
    def build_features():
        """
        Limpieza + feature engineering (Seccion 2 del TP de AMq1): dropea filas sin notas o
        accords completos, descarta el artefacto de scraping, deduplica, descarta clases
        chicas, arma el target y las 256 features (240 notas multi-hot + 6 numericas +
        dummies de concentracion). Guarda la matriz final y el diccionario de features
        (`data_info/data.json`) que usan tanto `split_dataset` como la API para transformar
        requests nuevos de la misma manera.

        La transformacion no se implementa aca: se importa de `src/perfume_features.py`, que
        es el mismo modulo que usan el notebook de entrenamiento y la API. Antes esta tarea
        tenia una copia en linea de esa logica, y cualquier cambio en una sola de las dos
        copias desincronizaba el pipeline del entrenamiento.
        """
        import json
        import sys
        import datetime as dt

        import boto3
        import mlflow
        import pandas as pd
        import awswrangler as wr
        from airflow.models import Variable

        # Las tareas @task.virtualenv corren en un venv aislado, que no tiene el repo en el
        # sys.path. El directorio src/ del repo se monta en el contenedor como
        # /opt/airflow/repo_src (ver `volumes` de x-airflow-common en docker-compose.yaml),
        # asi que basta con agregarlo al path para importar el modulo compartido.
        sys.path.insert(0, "/opt/airflow/repo_src")
        from perfume_features import build_gold_dataset, fit_feature_spec, transform

        min_samples = int(Variable.get("min_samples_per_class_perfume"))
        k_notes = int(Variable.get("k_notes_per_layer_perfume"))
        n_top_conc = int(Variable.get("n_top_concentrations_perfume"))
        min_year = int(Variable.get("min_valid_year_perfume"))

        df = wr.s3.read_csv("s3://data/raw/parfumo_perfumes.csv")

        gold = build_gold_dataset(
            df,
            min_samples_per_class=min_samples,
            min_valid_year=min_year,
        )
        spec = fit_feature_spec(
            gold,
            k_notes_per_layer=k_notes,
            n_top_concentrations=n_top_conc,
        )
        X, y, feature_names = transform(gold, spec)

        features_df = pd.DataFrame(X, columns=feature_names)
        features_df["dominant_accord"] = y

        wr.s3.to_csv(df=features_df, path="s3://data/final/parfumo_features.csv", index=False)

        # El spec es lo que la API necesita para transformar un request igual que en
        # entrenamiento; el resto son metadatos de esta corrida.
        data_dict = dict(spec)
        data_dict["n_rows"] = int(features_df.shape[0])
        data_dict["n_features"] = int(len(feature_names))
        data_dict["date"] = dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
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
            mlflow.log_param("n_classes", len(spec["classes"]))
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
        requirements=["awswrangler==3.6.0", "pyarrow==17.0.0"],
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
