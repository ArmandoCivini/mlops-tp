"""*Safe checks* del modelo que sirve la API.

La clase 1 lo planteó así: los modelos no son deterministas, así que las pruebas no verifican
una predicción exacta, verifican que la salida caiga en el rango aceptable antes de que llegue
a producción (el ejemplo del docente fue Mercado Pago ofreciendo un préstamo al 700% por una
predicción sin control de límites).

Acá se prueba el modelo de fallback empaquetado en la imagen de FastAPI, que es el que la API
usa cuando el registro de MLflow todavía no tiene un `champion`. Es el único modelo que está
versionado en el repo, así que es el único que se puede chequear sin levantar el stack.
"""

import numpy as np

from perfume_features import row_from_request, transform


def request_de_ejemplo():
    """El mismo ejemplo que documenta `/docs` y el README."""
    return row_from_request(
        top_notes=["Bergamot", "Lemon"],
        middle_notes=["Jasmine", "Rose"],
        base_notes=["Musk", "Sandalwood", "Vanilla"],
        release_year=2026,
        concentration="Eau de Parfum",
    )


def test_el_modelo_espera_la_misma_cantidad_de_features_que_el_data_json(
    packaged_model, api_spec
):
    """Si el `data.json` y el modelo se desincronizan, la API devuelve 500 en cada request.
    Es la "garantía de validez del modelo" de la clase 5: que el preprocesador y el modelo
    estén sincronizados."""
    assert packaged_model.n_features_in_ == api_spec["n_features"]


def test_el_modelo_conoce_las_mismas_clases_que_el_data_json(packaged_model, api_spec):
    assert sorted(str(clase) for clase in packaged_model.classes_) == sorted(api_spec["classes"])


def test_predice_una_familia_olfativa_conocida(packaged_model, api_spec):
    """El control de rango: la salida tiene que ser una de las 20 familias, nunca un valor
    inesperado."""
    X, _, _ = transform(request_de_ejemplo(), api_spec)

    prediccion = str(packaged_model.predict(X)[0])

    assert prediccion in api_spec["classes"]


def test_las_probabilidades_son_una_distribucion_valida(packaged_model, api_spec):
    """La API devuelve el top-3 de `predict_proba`. Si las probabilidades no suman 1 o se
    van de rango, lo que se le muestra al usuario no significa nada."""
    X, _, _ = transform(request_de_ejemplo(), api_spec)

    proba = packaged_model.predict_proba(X)[0]

    assert len(proba) == len(packaged_model.classes_)
    assert np.isclose(proba.sum(), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_la_prediccion_coincide_con_la_clase_mas_probable(packaged_model, api_spec):
    """Coherencia interna: lo que devuelve `predict` tiene que ser el argmax de
    `predict_proba`, que es lo que la API muestra como primera opción del top-3."""
    X, _, _ = transform(request_de_ejemplo(), api_spec)

    prediccion = str(packaged_model.predict(X)[0])
    mas_probable = str(packaged_model.classes_[np.argmax(packaged_model.predict_proba(X)[0])])

    assert prediccion == mas_probable


def test_es_determinista_para_el_mismo_request(packaged_model, api_spec):
    """Dos requests idénticos tienen que dar la misma respuesta: si no, no se puede ni
    reproducir un bug reportado."""
    X, _, _ = transform(request_de_ejemplo(), api_spec)

    assert packaged_model.predict(X)[0] == packaged_model.predict(X)[0]


def test_un_request_minimo_no_rompe_el_modelo(packaged_model, api_spec):
    """Todos los campos de `ModelInput` son opcionales. Un request casi vacío tiene que
    devolver una predicción válida, no una excepción."""
    row = row_from_request(top_notes=["Bergamot"], middle_notes=[], base_notes=[])

    X, _, _ = transform(row, api_spec)

    assert str(packaged_model.predict(X)[0]) in api_spec["classes"]
