# Changelog

Cambios del TP de MLOps I (clasificador de familia olfativa en perfumes) posteriores a la
**entrega parcial** corregida por la cátedra. El formato sigue, a grandes rasgos,
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [Fase 1 — verificación de punta a punta] — 2026-10-17

Se levantó el stack desde cero y se corrió todo el flujo: ETL (`process_perfume_data`), notebook
`experiment_mlflow.ipynb` (registro del campeón), DAG `retrain_the_model` y API (`/predict/`).
Al hacerlo aparecieron problemas de versiones que impedían correr el flujo; se corrigieron así.

### Corregido

- **`airflow/dags/etl_process.py` — `pyarrow` sin fijar.** La última versión de `pyarrow` exigía
  NumPy 2, incompatible con el NumPy 1.26.4 de la imagen de Airflow, y `get_data` fallaba. Se fijó
  `pyarrow==17.0.0` en las tareas del ETL.
- **`airflow/dags/retrain_the_model.py` — scikit-learn desalineado.** El venv de las tareas usaba
  scikit-learn 1.3.2 y el modelo se entrena (Jupyter) y se sirve (API) con 1.8.0. Se alineó a 1.8.0.
- **`airflow/dags/retrain_the_model.py` — campeón ilegible por diferencia de NumPy.** El campeón lo
  guarda Jupyter con NumPy 2.5.4; el venv heredaba el NumPy 1.26.4 de Airflow y
  `mlflow.sklearn.load_model` fallaba con `PCG64 is not a known BitGenerator module`. Se fijaron en
  el venv `numpy==2.5.4`, `pandas==2.3.3`, `pyarrow==20.0.0` y `awswrangler==3.12.1` (la 3.6.0 no
  resolvía junto con NumPy 2 y MLflow 3.17).
- **`airflow/dags/retrain_the_model.py` — registro de versiones con MLflow 3.17.1.** Con el servidor
  en 3.17.1, `create_model_version` rechaza fuentes `models:/m-...` si no se pasa `model_id`. La
  tarea de entrenamiento ahora devuelve `model_info.model_id` y `register_challenger` lo pasa.
  El cliente del venv se fijó en `mlflow==3.17.1`, igual que el servidor.

### Verificado

- ETL: 21.032 filas, 256 features, 20 clases.
- Notebook (`N_TRIALS=15`): campeón HistGradientBoosting, F1 macro de test 0.1754, registrado como
  `perfume_accord_model_prod` v1 con alias `champion`.
- `retrain_the_model`: ambas tareas en `success`. Challenger F1 macro ≈ 0.172 vs. campeón 0.1754;
  `Winner` = campeón, el alias `champion` queda en v1 y el alias `challenger` se elimina.
- API: `POST /predict/` devuelve 200 con `dominant_accord` y `top_probabilities`.

### Limitaciones conocidas

- No se ejercitó la rama de **promoción** (challenger que supera al campeón): en esta corrida ganó
  el campeón.
- El F1 es bajo (0.1754 con `N_TRIALS=15`, contra ≈0.194 con `N_TRIALS=30` en la corrida de
  septiembre); no se buscó mejorarlo en esta fase.
- Las versiones de MLflow (servidor y Jupyter) no están fijadas en los Dockerfiles; conviene
  fijarlas para evitar el mismo tipo de desfase.
- La búsqueda con Optuna necesita bastante memoria: con Docker limitado a ~8 GB terminó
  `SIGKILL (-9)`; funcionó subiendo el límite de memoria de Docker.

## [No publicado] — camino a la entrega final

Esta tanda responde, punto por punto, a la devolución de la entrega parcial.

### Agregado

- **Suite de tests automáticos con `pytest`** (31 tests, ~3 s, sin Docker), que cubre lo que la
  clase 1 pidió como pruebas contra un *golden dataset* y *safe checks* previos a producción:
  - `tests/test_perfume_features.py` — la transformación produce siempre las mismas 256 columnas
    en el mismo orden, sobre el dataset entero y sobre la única fila de un request de la API (el
    test que previene el *training-serving skew*), más el manejo de faltantes, concentraciones
    desconocidas y notas fuera del vocabulario.
  - `tests/test_etl_contract.py` — el contrato de la limpieza de la Sección 2 del TP de AMq1,
    incluyendo que `min_samples_per_class` y `k_notes_per_layer` (leídos de las Airflow
    Variables) tengan efecto real.
  - `tests/test_dataset_golden_numbers.py` — *golden numbers* del catálogo completo: 21.032
    filas limpias, 256 features, 20 clases.
  - `tests/test_model_safe_check.py` — *safe checks* del modelo empaquetado: features y clases
    coinciden con `data.json`, `predict_proba` es una distribución válida y `predict` coincide
    con el argmax.
- `requirements-dev.txt` con el entorno para correrlos fuera de Docker (Python 3.12 y
  `scikit-learn==1.8.0`, las mismas versiones que los contenedores) y sección **Tests** en el
  README con las instrucciones.

### Corregido

- **`transform()` hacía un downcast implícito deprecado en el camino de inferencia de la API**
  (bug destapado por los tests): con un request de una sola fila y `release_year` en `None`, la
  columna quedaba con dtype `object` y el `fillna` emitía un `FutureWarning` de pandas. Ahora
  las columnas numéricas pasan por `pd.to_numeric` antes de imputar; verificado que la matriz,
  el target y el `data.json` quedan idénticos sobre el CSV completo.

- **`retrain_the_model`: registro del challenger con el patrón roto de MLflow 3.x.** El DAG
  todavía usaba `mlflow.sklearn.log_model(..., artifact_path="model")` junto con
  `mlflow.get_artifact_uri(...)`, el mismo patrón que ya habíamos documentado como roto en
  `notebook_example/experiment_mlflow.ipynb`: con MLflow 3.x el modelo se guarda bajo el esquema
  de "Logged Models", así que esa URI reconstruida apunta a un archivo que no existe y la versión
  registrada queda con un `source` roto. Ahora se usa `name="model"` y la URI se toma de
  `model_info.model_uri`, igual que en el notebook. Como esa URI ya no contiene el run id,
  `register_challenger` lo recibe explícito desde `run.info.run_id`, en vez de parsearlo con
  `model_uri.split("/")[-3]`. El run del challenger además se abre con `with` (antes quedaba
  abierto) y loguea su propio `test_f1_macro`.
- **`retrain_the_model`: la evaluación escribía las métricas en un run ajeno.** La tarea
  `evaluate_champion_challenger` tomaba `mlflow.search_runs(...)[0]` —sin `order_by`, o sea un run
  arbitrario del experimento, que podía ser incluso un run del ETL— y le pegaba encima
  `test_f1_macro_champion`, `test_f1_macro_challenger` y el param `Winner`. Ahora cada evaluación
  abre su propio run, `Champion_vs_Challenger_<fecha>`, con su tag de experimento.

### Cambiado

- **Una sola implementación de la lógica de features.** La limpieza y el feature engineering
  estaban duplicados entre `src/perfume_features.py` (que usan el notebook y la API) y una copia
  en línea dentro de la tarea `build_features` del DAG de ETL. El DAG ahora importa el módulo
  compartido: `src/` se monta de solo lectura en los contenedores de Airflow como
  `/opt/airflow/repo_src` y la tarea lo agrega al `sys.path` (las tareas `@task.virtualenv` corren
  en un venv aislado, que era el motivo original de la duplicación).

  Para que el módulo cubra los tres usos:
  - `build_gold_dataset` y `fit_feature_spec` reciben `min_samples_per_class`, `min_valid_year`,
    `k_notes_per_layer` y `n_top_concentrations` como parámetros (los defaults son los valores del
    TP de AMq1). El DAG los sigue leyendo de las Airflow Variables y los pasa explícitos.
  - el `spec` incluye ahora `conc_categories` y `feature_names`, que antes armaba el DAG por su
    cuenta, así que `data_info/data.json` se serializa directo desde el `spec`.

  Verificado sobre `data/parfumo_perfumes.csv`: la matriz `X`, el target y el `data.json`
  resultante son **idénticos** a los de la lógica anterior (21.032 filas, 256 features, 20
  clases). La transformación de inferencia se probó además contra el `data.json` que lleva la
  imagen de FastAPI.

- **Una sola rama de entrega: `main`.** La entrega estaba partida: `main` tenía los diagramas de
  los DAGs y las reescrituras del README, y `tp-perfume-clasificador` tenía el fix de MLflow 3.x
  del notebook, la vuelta a MinIO con la imagen de `coollabsio`, los integrantes y el estado de
  avance verificado. Se mergeó la rama a `main`, que pasa a ser la única rama de entrega;
  `tp-perfume-clasificador` queda como historial de desarrollo.

### Documentación

- README: se declara explícitamente que la rama de entrega es `main` y se apunta a este
  CHANGELOG; se deja dicho que `src/perfume_features.py` es la única implementación de las
  features; se actualiza el estado del DAG de reentrenamiento y la nota del bug de MLflow 3.x para
  reflejar que el arreglo está también en el DAG.

---

## [Entrega parcial] — 2026-09-27

Estado tal como fue corregido por la cátedra.

### Agregado

- Stack completo en Docker: Airflow, MLflow, PostgreSQL, MinIO, Redis, Jupyter y FastAPI
  (`docker compose --profile all up --build`).
- DAG `process_perfume_data` (ETL): publicación del catálogo crudo de Parfumo en `s3://data`,
  limpieza, feature engineering y split estratificado 80/20. Corrido con las 3 tareas en
  `success`.
- `notebook_example/experiment_mlflow.ipynb`: búsqueda de hiperparámetros con Optuna sobre los 5
  algoritmos del TP de AMq1, con el mejor modelo registrado en MLflow con alias `champion`
  (`GradientBoosting`, `f1_macro` en test ≈ 0.194).
- Servicio FastAPI con `POST /predict/`, sirviendo el modelo `champion` real (con fallback a un
  modelo empaquetado en la imagen).
- DAG `retrain_the_model`: reentrenamiento y comparación challenger vs. campeón por `f1_macro`
  (implementado, sin correr).
- `src/perfume_features.py`, `docs/diagramas_dags.md` y README con el problema, el alcance y las
  instrucciones de puesta en marcha.

### Corregido

- `mlflow.get_artifact_uri()` con MLflow 3.x en el notebook de experimentación (el modelo quedaba
  registrado con un `source` roto y la API caía en silencio al modelo de fallback).
- Imagen de MinIO: MinIO borró sus imágenes públicas de Docker Hub (11-14/09/2026) y después
  bloqueó los tags fijos de `quay.io` (24/09/2026). Tras un parche temporal con SeaweedFS, se
  volvió a MinIO con `coollabsio/minio`, el mirror al que apunta el repo base de la cátedra.
- `pull_policy: build` en los servicios con imagen propia, para que `docker compose up` sin
  `--build` no intente bajarlas de Docker Hub.
- MLflow conectándose a Postgres con `postgresql+psycopg2://` explícito.
