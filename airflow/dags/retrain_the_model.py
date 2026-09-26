import datetime

from airflow.decorators import dag, task

markdown_text = """
### Re-Train the Perfume Olfactory-Family Model

Este DAG reentrena el modelo (clona la arquitectura del campeón actual, la reentrena sobre el
train set vigente en `s3://data`) y lo registra como `challenger`. Luego compara campeón vs.
challenger sobre el test set usando `f1_macro` (la misma métrica principal que se usó en el TP de
AMq1, elegida por el desbalance de las 20 clases) y promueve al challenger si es mejor.

Notar: antes de correr este DAG hace falta que exista un campeón en el registro de MLflow. Eso se
hace una única vez, a mano, corriendo `notebook_example/experiment_mlflow.ipynb` (ver README).

Nota sobre el corpus de test: correr `process_perfume_data` de nuevo antes de este DAG resamplea
el split 80/20 sobre el mismo dataset (no hay datos "nuevos" de Parfumo llegando en tiempo real),
así que es esperable que challenger y campeón midan parecido. Esto es una limitación reconocida en
el TP de AMq1 (ver conclusiones): en un caso real, la mejora vendría de nuevas fichas de perfumes
publicadas en Parfumo con el tiempo, no de resamplear el mismo catálogo.
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
    dag_id="retrain_the_model",
    description="Re-train the perfume olfactory-family model, and promote it to champion if it performs better.",
    doc_md=markdown_text,
    tags=["Re-Train", "Perfume Astral"],
    default_args=default_args,
    catchup=False,
)
def processing_dag():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["scikit-learn==1.3.2", "mlflow==3.1.4", "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def train_the_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr

        from sklearn.base import clone
        from sklearn.metrics import f1_score
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri("http://mlflow:5000")

        def load_the_champion_model():
            model_name = "perfume_accord_model_prod"
            alias = "champion"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            champion_version = mlflow.sklearn.load_model(model_data.source)

            return champion_version

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/perfume_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/perfume_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/perfume_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/perfume_y_test.csv")

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            experiment = mlflow.set_experiment("Perfume Olfactory Family")

            mlflow.start_run(
                run_name="Challenger_run_" + datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": "Parfumo perfumes"},
                log_system_metrics=True,
            )

            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            artifact_path = "model"
            signature = infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format="cloudpickle",
                registered_model_name="perfume_accord_model_dev",
                metadata={"model_data_version": 1},
            )

            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, f1_macro_score, model_uri):
            client = mlflow.MlflowClient()
            name = "perfume_accord_model_prod"

            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["f1-macro"] = f1_macro_score

            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags,
            )

            client.set_registered_model_alias(name, "challenger", result.version)

        # Cargamos el campeón actual y clonamos su arquitectura (mismos hiperparámetros,
        # sin entrenar) para el challenger.
        champion_model = load_the_champion_model()
        challenger_model = clone(champion_model)

        X_train, y_train, X_test, y_test = load_the_train_test_data()

        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        y_pred = challenger_model.predict(X_test)
        # f1_macro, no accuracy: con 20 clases desbalanceadas, accuracy sola esconde el
        # desempeño en las familias chicas (ver TP de AMq1, sección 4.1).
        f1_macro = f1_score(y_test.to_numpy().ravel(), y_pred, average="macro")

        artifact_uri = mlflow_track_experiment(challenger_model, X_train)
        register_challenger(challenger_model, f1_macro, artifact_uri)

    @task.virtualenv(
        task_id="evaluate_champion_challenger",
        requirements=["scikit-learn==1.3.2", "mlflow==3.1.4", "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def evaluate_champion_challenge():
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import f1_score

        mlflow.set_tracking_uri("http://mlflow:5000")

        def load_the_model(alias):
            model_name = "perfume_accord_model_prod"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            return mlflow.sklearn.load_model(model_data.source)

        def load_the_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/perfume_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/perfume_y_test.csv")

            return X_test, y_test

        def promote_challenger(name):
            client = mlflow.MlflowClient()

            client.delete_registered_model_alias(name, "champion")

            challenger_version = client.get_model_version_by_alias(name, "challenger")
            client.delete_registered_model_alias(name, "challenger")

            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):
            client = mlflow.MlflowClient()
            client.delete_registered_model_alias(name, "challenger")

        champion_model = load_the_model("champion")
        challenger_model = load_the_model("challenger")

        X_test, y_test = load_the_test_data()

        y_pred_champion = champion_model.predict(X_test)
        f1_macro_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion, average="macro")

        y_pred_challenger = challenger_model.predict(X_test)
        f1_macro_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger, average="macro")

        experiment = mlflow.set_experiment("Perfume Olfactory Family")
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_f1_macro_challenger", f1_macro_challenger)
            mlflow.log_metric("test_f1_macro_champion", f1_macro_champion)

            if f1_macro_challenger > f1_macro_champion:
                mlflow.log_param("Winner", "Challenger")
            else:
                mlflow.log_param("Winner", "Champion")

        name = "perfume_accord_model_prod"
        if f1_macro_challenger > f1_macro_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
