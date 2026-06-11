import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para trabajar en segundo plano
import matplotlib.pyplot as plt
from toolbox_ml.eda.core import describe_df, tipifica_variables

#### --- TEST FUNCIÓN DESCRIBE_DF ----

# 1. TEST

def test_describe_df_devuelve_dataframe():
    """
    Caso correcto: input válido → retorna DataFrame.
    """

    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame) # Comprobación de que es un DF


# 2. TEST

def test_describe_df_devuelve_dataframe_vacio():
    """
    Caso límite: DataFrame vacío → debe devolver un DataFrame vacío.
    """
    
    df = pd.DataFrame() # df vacío
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame) # Comprobación de que es un DF
    assert resultado.empty # DF vacío


# 3. TEST

def test_describe_df_devuelve_dataframe_input_invalido():
    """
    Caso de error: Si el input no es un DataFrame → devuelve None.
    """
    df = "x" # no es un df
    resultado = describe_df(df)
    assert resultado is None


# 4. TEST

def test_describe_df_columnas_correctas():
    """
    El DataFrame resultado tiene exactamente las columnas esperadas.
    """

    df = pd.DataFrame({'a': [1, 2, 3]})
    resultado = describe_df(df)
    assert set(resultado.columns) == {
        'tipo', 'porcentaje_nulos', 'valores_unicos', 'porcentaje_cardinalidad'
    }

# 5. TEST

def test_describe_df_porcentaje_nulos_correcto():
    """
    Calcula correctamente el porcentaje de nulos.
    """

    df = pd.DataFrame({'a': [1, None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc['a', 'porcentaje_nulos'] == 75.0


#### --- TEST FUNCIÓN TIPIFICA VARIABLES --- 

# 1. TEST

def test_tipifica_variables_devuelve_dataframe():
    """
    Caso correcto: input válido → retorna DataFrame.
    """

    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = tipifica_variables(df, umbral_categorica=10, umbral_continua=0.3)
    assert isinstance(resultado, pd.DataFrame) # Devuelve DF


# 2. TEST

def test_tipifica_variables_dataframe_vacio():
    """
    Caso límite: DataFrame vacío → debe devolver un DataFrame vacío.
    """
    df = pd.DataFrame()  # sin columnas
    resultado = tipifica_variables(df, 10, 5.0)
    assert isinstance(resultado, pd.DataFrame) # devuelve DF
    assert resultado.empty # DF vacío

# 3. TEST

def test_tipifica_variables_input_invalido():
    """
    Caso de error: Si el input no es un DataFrame → devuelve None.
    """
    resultado = tipifica_variables("X", 10, 5.0)
    assert resultado is None

# 4. TEST

def test_tipifica_variables_clasifica_categorica_correctamente():
    """
    Comprueba que una variable con cardinalidad menor que el umbral
    se clasifica como 'Categórica'.
    """
    df = pd.DataFrame({'a': [1, 2, 3, 4]}) # DF con 4 valores únicos
    resultado = tipifica_variables(df, 10, 5.0)
    assert resultado.loc[0, 'tipo_sugerido'] == "Categórica"

# 5. TEST

def test_tipifica_variables_clasifica_binaria_correctamente():
    """
    Comprueba que una variable con cardinalidad = 2
    se clasifica como 'Binaria'.
    """
    df = pd.DataFrame({'a': [1, 2]}) # DF con 2 valores únicos
    resultado = tipifica_variables(df, 10, 5.0)
    assert resultado.loc[0, 'tipo_sugerido'] == "Binaria"

"""
Tests de las funciones de Claudia (Desarrolladora 2).
3 tests por función: caso correcto, caso límite y caso de error.
Ejecutar desde la raíz:  pytest tests/ -v
"""

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
    plt.close('all')  # <--- Añado para limpiar la memoria


def test_plot_num_columns_vacia_usa_todas(df_num):
    """Caso límite: columns vacía -> usa todas las numéricas como candidatas."""
    res = plot_features_num_regression(df_num, "y", columns=[], umbral_corr=0.9)
    assert "x_buena" in res
    plt.close('all')  # <--- Añado para limpiar la memoria


def test_plot_num_caso_error(df_num):
    """Caso de error: target no numérico -> None."""
    df = df_num.copy()
    df["texto"] = ["a", "b", "c", "d", "e", "f"]
    assert plot_features_num_regression(df, "texto", umbral_corr=0.5) is None
    plt.close('all')  # <--- Añado para limpiar la memoria


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
    plt.close('all')  # <--- Añado para limpiar la memoria

def test_plot_cat_individual(df_cat):
    """Caso límite: with_individual_plot=True también funciona."""
    res = plot_features_cat_regression(df_cat, "precio", with_individual_plot=True)
    assert isinstance(res, list)
    plt.close('all')  # <--- Añado para limpiar la memoria


def test_plot_cat_caso_error(df_cat):
    """Caso de error: df inválido -> None."""
    assert plot_features_cat_regression("no es df", "precio") is None
    plt.close('all')  # <--- Añado para limpiar la memoria


