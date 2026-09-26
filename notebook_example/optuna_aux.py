import mlflow

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def objective(trial, X_train, y_train, experiment_id):
    """
    Optimiza hiperparámetros para el clasificador de familia olfativa usando Optuna.

    A diferencia del ejemplo de Heart Disease (clasificación binaria, dataset chico), acá
    tenemos 20 clases desbalanceadas y ~16.800 filas x 256 features, así que:

    - La métrica de selección es `f1_macro` (no accuracy ni f1 binario): promedia el F1 de
      cada una de las 20 familias por igual, en línea con el criterio del TP de AMq1
      (sección 4.1), que muestra que accuracy sola queda dominada por las clases grandes.
    - Los algoritmos elegidos son los mismos 5 que se compararon en el TP de AMq1 (KNN,
      Árbol de decisión, Random Forest, Gradient Boosting, SVM), pero el kernel de SVM se
      restringe a RBF: el propio TP de AMq1 encontró que el kernel lineal tenía "tiempo de
      entrenamiento excesivo" en este dataset y fue descartado (sección 3.2), y acá el
      espacio de búsqueda ya es más pesado que el ejemplo de Heart Disease por el tamaño del
      dataset.

    Parameters
    ----------
    trial : optuna.trial.Trial
    X_train : pandas.DataFrame
    y_train : pandas.Series o pandas.DataFrame de una columna
    experiment_id : int
        ID del experimento de MLflow donde se van a loguear los resultados.

    Returns
    -------
    float
        f1_macro promedio de validación cruzada.
    """

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Trial: {trial.number}", nested=True):

        params = {"objective": "multiclass classification", "eval_metric": "f1_macro"}

        classifier_name = trial.suggest_categorical(
            "classifier",
            ["KNN", "DecisionTree", "RandomForest", "GradientBoosting", "SVC_rbf"],
        )

        if classifier_name == "KNN":
            n_neighbors = trial.suggest_int("knn_n_neighbors", 5, 35, step=2)
            weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])

            classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

            params["model"] = "KNeighborsClassifier"
            params["n_neighbors"] = n_neighbors
            params["weights"] = weights

        elif classifier_name == "DecisionTree":
            max_depth = trial.suggest_int("tree_max_depth", 4, 32, log=True)
            min_samples_leaf = trial.suggest_int("tree_min_samples_leaf", 1, 20)

            classifier_obj = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

            params["model"] = "DecisionTreeClassifier"
            params["max_depth"] = max_depth
            params["min_samples_leaf"] = min_samples_leaf

        elif classifier_name == "RandomForest":
            n_estimators = trial.suggest_int("rf_n_estimators", 100, 400, step=50)
            max_depth = trial.suggest_int("rf_max_depth", 6, 32, log=True)
            min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)

            classifier_obj = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
            )

            params["model"] = "RandomForestClassifier"
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
            params["min_samples_leaf"] = min_samples_leaf

        elif classifier_name == "GradientBoosting":
            learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
            max_depth_choice = trial.suggest_categorical("gb_max_depth", [None, 6, 10])
            l2_regularization = trial.suggest_float("gb_l2_regularization", 0.0, 1.5)

            classifier_obj = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth_choice,
                l2_regularization=l2_regularization,
                max_iter=300,
                early_stopping=True,
            )

            params["model"] = "HistGradientBoostingClassifier"
            params["learning_rate"] = learning_rate
            params["max_depth"] = max_depth_choice
            params["l2_regularization"] = l2_regularization

        else:
            # SVC con kernel RBF. Ver docstring: el kernel lineal se descarta a propósito.
            svc_c = trial.suggest_float("svc_c", 0.1, 50, log=True)

            classifier_obj = SVC(C=svc_c, kernel="rbf", gamma="scale")

            params["model"] = "SVC"
            params["kernel"] = "rbf"
            params["C"] = svc_c

        # cv=3 (no 5) y n_trials moderado (ver notebook): con ~16.800 filas x 256 features,
        # un espacio de búsqueda tan amplio como el del ejemplo de Heart Disease (250 trials,
        # cv=5) sería impracticable en una laptop. SVC es, a propósito, el candidato más
        # lento de esta grilla.
        score = cross_val_score(
            classifier_obj, X_train, y_train.to_numpy().ravel(), n_jobs=-1, cv=3, scoring="f1_macro"
        )

        mlflow.log_params(params)
        mlflow.log_metric("f1_macro", score.mean())

    return score.mean()
