"""
Tests de las funciones de Claudia (Desarrolladora 2).
3 tests por función: caso correcto, caso límite y caso de error.
Ejecutar desde la raíz:  pytest tests/ -v
"""
import pytest
import pandas as pd

from toolbox_ml.eda.core import (
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression,
)


@pytest.fixture
def df_num():
    """Tabla de prueba: 'y' es el target; 'x_buena' correlaciona, 'x_ruido' no."""
    return pd.DataFrame({
        "y":       [1, 2, 3, 4, 5, 6],
        "x_buena": [2, 4, 6, 8, 10, 12],   # correlación perfecta con y
        "x_ruido": [5, 1, 4, 2, 6, 3],     # sin relación
    })


@pytest.fixture
def df_cat():
    """El precio depende mucho del grupo 'clase'; 'color' es irrelevante."""
    return pd.DataFrame({
        "precio": [10, 12, 11, 90, 95, 88, 50, 52, 49],
        "clase":  ["baja", "baja", "baja", "alta", "alta", "alta", "media", "media", "media"],
        "color":  ["rojo", "azul", "rojo", "azul", "rojo", "azul", "rojo", "azul", "rojo"],
    })


# ----------------------------- get_features_num_regression -----------------------------
def test_num_caso_correcto(df_num):
    """Caso correcto: la columna que correlaciona aparece y el ruido no."""
    res = get_features_num_regression(df_num, "y", umbral_corr=0.9)
    assert "x_buena" in res
    assert "x_ruido" not in res


def test_num_caso_limite_umbral_imposible(df_num):
    """Caso límite: con umbral por encima de 1 nada lo supera -> lista vacía."""
    assert get_features_num_regression(df_num, "y", umbral_corr=0.999999) == ["x_buena"]
    assert get_features_num_regression(df_num, "y", umbral_corr=1) == []


def test_num_caso_error_devuelve_none(df_num):
    """Caso de error: entradas inválidas -> None."""
    assert get_features_num_regression("no es df", "y", 0.5) is None
    assert get_features_num_regression(df_num, "no_existe", 0.5) is None
    assert get_features_num_regression(df_num, "y", 5) is None  # umbral fuera de [0,1]


# ----------------------------- plot_features_num_regression -----------------------------
def test_plot_num_devuelve_lista(df_num):
    """Caso correcto: devuelve la lista de columnas pintadas."""
    res = plot_features_num_regression(df_num, "y", umbral_corr=0.9)
    assert isinstance(res, list)
    assert "x_buena" in res


def test_plot_num_columns_vacia_usa_todas(df_num):
    """Caso límite: columns vacía -> usa todas las numéricas como candidatas."""
    res = plot_features_num_regression(df_num, "y", columns=[], umbral_corr=0.9)
    assert "x_buena" in res


def test_plot_num_caso_error(df_num):
    """Caso de error: target no numérico -> None."""
    df = df_num.copy()
    df["texto"] = ["a", "b", "c", "d", "e", "f"]
    assert plot_features_num_regression(df, "texto", umbral_corr=0.5) is None


# ----------------------------- get_features_cat_regression -----------------------------
def test_cat_caso_correcto(df_cat):
    """Caso correcto: 'clase' (3 grupos, ANOVA) influye y aparece."""
    res = get_features_cat_regression(df_cat, "precio", pvalue=0.05)
    assert "clase" in res


def test_cat_caso_limite_pvalue_estricto(df_cat):
    """Caso límite: con pvalue muy estricto, 'color' (irrelevante) no entra."""
    res = get_features_cat_regression(df_cat, "precio", pvalue=0.0001)
    assert "color" not in res


def test_cat_caso_error_devuelve_none(df_cat):
    """Caso de error: target categórico o pvalue inválido -> None."""
    assert get_features_cat_regression(df_cat, "clase", pvalue=0.05) is None  # target no numérico
    assert get_features_cat_regression(df_cat, "precio", pvalue=2) is None    # pvalue fuera de [0,1]


# ----------------------------- plot_features_cat_regression -----------------------------
def test_plot_cat_devuelve_lista(df_cat):
    """Caso correcto: devuelve lista con las categóricas pintadas."""
    res = plot_features_cat_regression(df_cat, "precio", pvalue=0.05)
    assert isinstance(res, list)
    assert "clase" in res


def test_plot_cat_individual(df_cat):
    """Caso límite: with_individual_plot=True también funciona."""
    res = plot_features_cat_regression(df_cat, "precio", with_individual_plot=True)
    assert isinstance(res, list)


def test_plot_cat_caso_error(df_cat):
    """Caso de error: df inválido -> None."""
    assert plot_features_cat_regression("no es df", "precio") is None
