# Diagramas de los DAGs de Airflow

Estos diagramas describen las dependencias implementadas actualmente en `airflow/dags/`.
Están escritos en Mermaid, por lo que GitHub los renderiza directamente y también pueden
editarse o exportarse desde el editor de Mermaid.

## `process_perfume_data`: ETL y partición del dataset

```mermaid
flowchart LR
    inicio((Inicio manual)) --> obtener

    subgraph Airflow[Apache Airflow · DAG process_perfume_data]
        obtener["get_data<br/>Lee el CSV local de Parfumo"]
        construir["build_features<br/>Limpia datos y construye features"]
        dividir["split_dataset<br/>Split estratificado 80/20"]
        obtener --> construir --> dividir
    end

    csv[("data/parfumo_perfumes.csv<br/>59.325 perfumes")]
    s3raw[("MinIO / S3<br/>data/raw/parfumo_perfumes.csv")]
    s3features[("MinIO / S3<br/>data/final/parfumo_features.csv")]
    especificacion[("MinIO / S3<br/>data/data_info/data.json")]
    train[("MinIO / S3<br/>data/final/train/<br/>X_train · y_train")]
    test[("MinIO / S3<br/>data/final/test/<br/>X_test · y_test")]
    mlflow[("MLflow<br/>Experimento: Perfume Olfactory Family<br/>metadata del dataset")]

    csv --> obtener
    obtener --> s3raw
    s3raw --> construir
    construir --> s3features
    construir --> especificacion
    construir --> mlflow
    s3features --> dividir
    dividir --> train
    dividir --> test

    classDef airflow fill:#e7f0ff,stroke:#2b6cb0,color:#1a365d;
    classDef storage fill:#ecfdf5,stroke:#2f855a,color:#22543d;
    classDef external fill:#fffaf0,stroke:#b7791f,color:#744210;
    class obtener,construir,dividir airflow;
    class s3raw,s3features,especificacion,train,test storage;
    class csv,mlflow external;
```

`build_features` conserva perfumes con notas y accords completos, quita duplicados y
clases pequeñas, y crea 256 variables: indicadores de notas, variables numéricas y
categorías de concentración. `data.json` almacena los vocabularios y parámetros de
transformación que después necesita la API para construir las mismas columnas.

## `retrain_the_model`: challenger contra campeón

```mermaid
flowchart TB
    inicio((Inicio manual)) --> requisito
    requisito{"¿Existe un modelo<br/>con alias champion?"}
    requisito -- No --> detener["Detener: primero ejecutar<br/>experiment_mlflow.ipynb"]
    requisito -- Sí --> entrenar

    subgraph Airflow[Apache Airflow · DAG retrain_the_model]
        entrenar["train_the_challenger_model<br/>1. Carga y clona el champion<br/>2. Entrena con X_train / y_train<br/>3. Registra un challenger"]
        evaluar["evaluate_champion_challenger<br/>1. Evalúa ambos en test<br/>2. Compara f1_macro"]
        entrenar --> evaluar
    end

    champion_actual[("MLflow Model Registry<br/>Entrada: champion actual")]
    datos_train[("MinIO / S3<br/>X_train · y_train")]
    datos_test[("MinIO / S3<br/>X_test · y_test")]
    experimento[("MLflow<br/>métricas, parámetros y artefacto")]
    challenger[("MLflow Model Registry<br/>Salida intermedia: challenger")]
    decision{"¿f1_macro del challenger<br/>es mayor que el champion?"}
    promover["Promueve la nueva versión<br/>como champion"]
    descartar["Descarta el challenger y conserva<br/>el champion actual"]
    registro_final[("MLflow Model Registry<br/>Estado final: champion vigente")]
    api["FastAPI<br/>En una predicción posterior detecta<br/>un champion nuevo y lo recarga"]

    champion_actual --> entrenar
    datos_train --> entrenar
    entrenar --> experimento
    entrenar --> challenger
    champion_actual --> evaluar
    challenger --> evaluar
    datos_test --> evaluar
    evaluar --> decision
    decision -- Sí --> promover
    decision -- No --> descartar
    promover --> registro_final
    descartar --> registro_final
    registro_final --> api

    classDef airflow fill:#e7f0ff,stroke:#2b6cb0,color:#1a365d;
    classDef storage fill:#ecfdf5,stroke:#2f855a,color:#22543d;
    classDef registry fill:#f3e8ff,stroke:#805ad5,color:#44337a;
    classDef decision fill:#fffaf0,stroke:#b7791f,color:#744210;
    class entrenar,evaluar airflow;
    class datos_train,datos_test storage;
    class champion_actual,challenger,experimento,registro_final registry;
    class requisito,decision decision;
```

El DAG no hace búsqueda de hiperparámetros: reutiliza la arquitectura e hiperparámetros
del campeón actual, la reentrena sobre el split vigente y decide la promoción usando
`f1_macro`. La búsqueda inicial y el primer registro del campeón se hacen en
`notebook_example/experiment_mlflow.ipynb`.
