"""Tests de `src/perfume_features.py`.

El punto de estos tests es evitar el *training-serving skew*: que la transformación que se
aplica en entrenamiento (DAG de ETL y notebook) y la que se aplica en inferencia (API) no
puedan divergir sin que algo se ponga rojo. Es la misma idea que la clase 5 planteó como
buena práctica ("evitar diferencias entre el código de entrenamiento y el de servicio").
"""

import numpy as np

from perfume_features import (
    conc_categories,
    feature_names_from_spec,
    row_from_request,
    transform,
)

N_FEATURES_ESPERADAS = 256


def columna(feature_names: list, nombre: str) -> int:
    """Índice de una feature por nombre, para no depender del orden a mano."""
    return feature_names.index(nombre)


def test_transform_devuelve_las_mismas_columnas_para_cualquier_cantidad_de_filas(
    gold_sample, sample_spec
):
    """El mismo spec tiene que producir las mismas columnas, en el mismo orden, ya sea que
    se transformen 2.000 filas o una sola. Si no, el modelo entrenado con la matriz grande
    recibe en producción un vector con las columnas corridas."""
    X_todo, _, names_todo = transform(gold_sample, sample_spec)
    X_pocas, _, names_pocas = transform(gold_sample.head(5), sample_spec)
    X_una, _, names_una = transform(gold_sample.head(1), sample_spec)

    assert names_todo == names_pocas == names_una
    assert X_todo.shape[1] == X_pocas.shape[1] == X_una.shape[1] == len(names_todo)
    assert X_pocas.shape[0] == 5
    assert X_una.shape[0] == 1


def test_el_spec_declara_las_mismas_columnas_que_devuelve_transform(gold_sample, sample_spec):
    """`feature_names` del spec es lo que se serializa a `data.json` y lo que la API usa para
    documentarse: tiene que coincidir con lo que realmente sale de `transform`."""
    _, _, names = transform(gold_sample, sample_spec)

    assert names == sample_spec["feature_names"]
    assert names == feature_names_from_spec(sample_spec)
    assert len(names) == len(set(names)), "hay nombres de features repetidos"


def test_la_matriz_no_tiene_nulos(gold_sample, sample_spec):
    """Los faltantes se imputan dentro de `transform` (año y n_accords). Si queda un NaN,
    el modelo falla en producción con un request perfectamente válido."""
    X, _, _ = transform(gold_sample, sample_spec)

    assert not np.isnan(X).any()


def test_inferencia_de_una_fila_coincide_con_el_data_json_de_la_api(api_spec):
    """El camino exacto de la API: `row_from_request` + `transform` contra el spec que
    viaja en la imagen. Las 256 columnas tienen que salir en el orden del `data.json`."""
    row = row_from_request(
        top_notes=["Bergamot", "Lemon"],
        middle_notes=["Jasmine", "Rose"],
        base_notes=["Musk", "Sandalwood", "Vanilla"],
        release_year=2026,
        concentration="Eau de Parfum",
    )

    X, y, names = transform(row, api_spec)

    assert X.shape == (1, N_FEATURES_ESPERADAS)
    assert X.shape[1] == api_spec["n_features"]
    assert names == api_spec["feature_names"]
    assert y is None, "en inferencia no hay target"


def test_una_concentracion_nunca_vista_cae_en_otras(api_spec):
    """Una categoría fuera del top-N de entrenamiento no puede generar una columna nueva:
    tiene que caer en `conc_Otras`."""
    row = row_from_request(
        top_notes=["Bergamot"],
        middle_notes=["Rose"],
        base_notes=["Musk"],
        concentration="Concentración Inventada Que No Existe",
    )

    X, _, names = transform(row, api_spec)

    assert X[0, columna(names, "conc_Otras")] == 1
    assert X[0, columna(names, "conc_Unknown")] == 0


def test_una_concentracion_faltante_cae_en_unknown(api_spec):
    """Distinto de `Otras`: acá el dato no vino, no es que vino raro."""
    row = row_from_request(
        top_notes=["Bergamot"],
        middle_notes=["Rose"],
        base_notes=["Musk"],
        concentration=None,
    )

    X, _, names = transform(row, api_spec)

    assert X[0, columna(names, "conc_Unknown")] == 1
    assert X[0, columna(names, "conc_Otras")] == 0


def test_exactamente_una_columna_de_concentracion_esta_activa(api_spec):
    """Las dummies de concentración son mutuamente excluyentes."""
    row = row_from_request(
        top_notes=["Bergamot"],
        middle_notes=["Rose"],
        base_notes=["Musk"],
        concentration="Eau de Parfum",
    )

    X, _, names = transform(row, api_spec)
    columnas_conc = [i for i, nombre in enumerate(names) if nombre.startswith("conc_")]

    assert X[0, columnas_conc].sum() == 1
    assert len(columnas_conc) == len(conc_categories(api_spec["top_concentrations"]))


def test_una_nota_fuera_del_vocabulario_se_ignora(api_spec):
    """El vocabulario es el top-K de entrenamiento. Una nota desconocida no puede romper
    nada ni activar una columna ajena: simplemente no aporta."""
    row = row_from_request(
        top_notes=["Nota Que No Existe En Parfumo"],
        middle_notes=[],
        base_notes=[],
    )

    X, _, names = transform(row, api_spec)
    columnas_notas = [
        i
        for i, nombre in enumerate(names)
        if nombre.startswith(("top__", "mid__", "base__"))
    ]

    assert X[0, columnas_notas].sum() == 0


def test_el_anio_faltante_se_imputa_con_la_mediana_de_entrenamiento(api_spec):
    """Documentado en el README como decisión, no como bug: si no viene el año, se usa la
    mediana de train en vez de rechazar el request."""
    row = row_from_request(
        top_notes=["Bergamot"],
        middle_notes=["Rose"],
        base_notes=["Musk"],
        release_year=None,
    )

    X, _, names = transform(row, api_spec)

    assert X[0, columna(names, "Release_Year")] == api_spec["year_median"]


def test_n_accords_se_imputa_porque_en_inferencia_no_existe(api_spec):
    """`n_accords` sale del mismo campo que el target, así que para un perfume nuevo no está
    disponible. `row_from_request` lo deja en NaN y `transform` lo imputa."""
    row = row_from_request(
        top_notes=["Bergamot"],
        middle_notes=["Rose"],
        base_notes=["Musk"],
    )

    X, _, names = transform(row, api_spec)

    assert X[0, columna(names, "n_accords")] == api_spec["n_accords_median"]


def test_las_cuentas_de_notas_son_la_cantidad_de_notas_del_request(api_spec):
    """Las features numéricas de conteo tienen que reflejar el request, no el vocabulario:
    acá hay 5 notas en total y solo una está en el top-K."""
    row = row_from_request(
        top_notes=["Bergamot", "Nota Inexistente"],
        middle_notes=["Rose"],
        base_notes=["Musk", "Otra Nota Inexistente"],
    )

    X, _, names = transform(row, api_spec)

    assert X[0, columna(names, "n_top")] == 2
    assert X[0, columna(names, "n_mid")] == 1
    assert X[0, columna(names, "n_base")] == 2
    assert X[0, columna(names, "n_notas_total")] == 5
