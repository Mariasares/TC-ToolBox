import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para trabajar en segundo plano
import matplotlib.pyplot as plt
from toolbox_ml.eda.core import (
    describe_df, 
    tipifica_variables, 
    get_features_num_regression, 
    plot_features_num_regression, 
    get_features_cat_regression, 
    plot_features_cat_regression, 
    get_features_cat_classification, 
    get_features_num_classification
)



# ----------------------------- describe_df -----------------------------

def test_describe_df_devuelve_dataframe():
    """
    Caso correcto: input válido → retorna DataFrame.
    """

    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame) # Comprobación de que es un DF


def test_describe_df_devuelve_dataframe_vacio():
    """
    Caso límite: DataFrame vacío → debe devolver un DataFrame vacío.
    """
    
    df = pd.DataFrame() # df vacío
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame) # Comprobación de que es un DF
    assert resultado.empty # DF vacío


def test_describe_df_devuelve_dataframe_input_invalido():
    """
    Caso de error: Si el input no es un DataFrame → devuelve None.
    """
    df = "x" # no es un df
    resultado = describe_df(df)
    assert resultado is None


def test_describe_df_columnas_correctas():
    """
    El DataFrame resultado tiene exactamente las columnas esperadas.
    """
    df = pd.DataFrame({'a': [1, 2, 3]})
    resultado = describe_df(df)
    assert set(resultado.columns) == {
        'tipo', 'porcentaje_nulos', 'valores_unicos', 'porcentaje_cardinalidad'
    }


def test_describe_df_porcentaje_nulos_correcto():
    """
    Calcula correctamente el porcentaje de nulos.
    """

    df = pd.DataFrame({'a': [1, None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc['a', 'porcentaje_nulos'] == 75.0


# ----------------------------- tipifica_variables -----------------------------

def test_tipifica_variables_devuelve_dataframe():
    """
    Caso correcto: input válido → retorna DataFrame.
    """

    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = tipifica_variables(df, umbral_categorica=10, umbral_continua=0.3)
    assert isinstance(resultado, pd.DataFrame) # Devuelve DF


def test_tipifica_variables_dataframe_vacio():
    """
    Caso límite: DataFrame vacío → debe devolver un DataFrame vacío.
    """
    df = pd.DataFrame()  # sin columnas
    resultado = tipifica_variables(df, 10, 5.0)
    assert isinstance(resultado, pd.DataFrame) # devuelve DF
    assert resultado.empty # DF vacío


def test_tipifica_variables_input_invalido():
    """
    Caso de error: Si el input no es un DataFrame → devuelve None.
    """
    resultado = tipifica_variables("X", 10, 5.0)
    assert resultado is None


def test_tipifica_variables_clasifica_categorica_correctamente():
    """
    Comprueba que una variable con cardinalidad menor que el umbral
    se clasifica como 'Categórica'.
    """
    df = pd.DataFrame({'a': [1, 2, 3, 4]}) # DF con 4 valores únicos
    resultado = tipifica_variables(df, 10, 5.0)
    assert resultado.loc[0, 'tipo_sugerido'] == "Categórica"


def test_tipifica_variables_clasifica_binaria_correctamente():
    """
    Comprueba que una variable con cardinalidad = 2
    se clasifica como 'Binaria'.
    """
    df = pd.DataFrame({'a': [1, 2]}) # DF con 2 valores únicos
    resultado = tipifica_variables(df, 10, 5.0)
    assert resultado.loc[0, 'tipo_sugerido'] == "Binaria"


# ----------------------------- Dataset numérico para pruebas de regresión -----------------------------

@pytest.fixture
def df_num_reg():
    """Tabla de prueba: 'y' es el target; 'x_buena' correlaciona, 'x_ruido' no."""
    return pd.DataFrame({
        "y":       [1, 2, 3, 4, 5, 6],
        "x_buena": [2, 4, 6, 8, 10, 12],   # correlación perfecta con y
        "x_ruido": [5, 1, 4, 2, 6, 3],     # sin relación
    })


# ----------------------------- Dataset categórico para pruebas de regresión -----------------------------

@pytest.fixture
def df_cat_reg():
    """El precio depende mucho del grupo 'clase'; 'color' es irrelevante."""
    return pd.DataFrame({
        "precio": [10, 12, 11, 90, 95, 88, 50, 52, 49],
        "clase":  ["baja", "baja", "baja", "alta", "alta", "alta", "media", "media", "media"],
        "color":  ["rojo", "azul", "rojo", "azul", "rojo", "azul", "rojo", "azul", "rojo"],
    })


# ----------------------------- get_features_num_regression -----------------------------

def test_num_caso_correcto(df_num_reg):
    """Caso correcto: la columna que correlaciona aparece y el ruido no."""
    res = get_features_num_regression(df_num_reg, "y", umbral_corr=0.9)
    assert "x_buena" in res
    assert "x_ruido" not in res


def test_num_caso_limite_umbral_imposible(df_num_reg):
    """Caso límite: con umbral por encima de 1 nada lo supera -> lista vacía."""
    assert get_features_num_regression(df_num_reg, "y", umbral_corr=0.999999) == ["x_buena"]
    assert get_features_num_regression(df_num_reg, "y", umbral_corr=1) == []


def test_num_caso_error_devuelve_none(df_num_reg):
    """Caso de error: entradas inválidas -> None."""
    assert get_features_num_regression("no es df", "y", 0.5) is None
    assert get_features_num_regression(df_num_reg, "no_existe", 0.5) is None
    assert get_features_num_regression(df_num_reg, "y", 5) is None  # umbral fuera de [0,1]


# ----------------------------- plot_features_num_regression -----------------------------

def test_plot_num_devuelve_lista(df_num_reg):
    """Caso correcto: devuelve la lista de columnas pintadas."""
    res = plot_features_num_regression(df_num_reg, "y", umbral_corr=0.9)
    assert isinstance(res, list)
    assert "x_buena" in res
    plt.close('all')


def test_plot_num_columns_vacia_usa_todas(df_num_reg):
    """Caso límite: columns vacía -> usa todas las numéricas como candidatas."""
    res = plot_features_num_regression(df_num_reg, "y", columns=[], umbral_corr=0.9)
    assert "x_buena" in res
    plt.close('all')


def test_plot_num_caso_error(df_num_reg):
    """Caso de error: target no numérico -> None."""
    df = df_num_reg.copy()
    df["texto"] = ["a", "b", "c", "d", "e", "f"]
    assert plot_features_num_regression(df, "texto", umbral_corr=0.5) is None
    plt.close('all')


# ----------------------------- get_features_cat_regression -----------------------------

def test_cat_caso_correcto(df_cat_reg):
    """Caso correcto: 'clase' (3 grupos, ANOVA) influye y aparece."""
    res = get_features_cat_regression(df_cat_reg, "precio", pvalue=0.05)
    assert "clase" in res


def test_cat_caso_limite_pvalue_estricto(df_cat_reg):
    """Caso límite: con pvalue muy estricto, 'color' (irrelevante) no entra."""
    res = get_features_cat_regression(df_cat_reg, "precio", pvalue=0.0001)
    assert "color" not in res


def test_cat_caso_error_devuelve_none(df_cat_reg):
    """Caso de error: target categórico o pvalue inválido -> None."""
    assert get_features_cat_regression(df_cat_reg, "clase", pvalue=0.05) is None  # target no numérico
    assert get_features_cat_regression(df_cat_reg, "precio", pvalue=2) is None    # pvalue fuera de [0,1]


# ----------------------------- plot_features_cat_regression -----------------------------

def test_plot_cat_devuelve_lista(df_cat_reg):
    """Caso correcto: devuelve lista con las categóricas pintadas."""
    res = plot_features_cat_regression(df_cat_reg, "precio", pvalue=0.05)
    assert isinstance(res, list)
    assert "clase" in res
    plt.close('all')


def test_plot_cat_individual(df_cat_reg):
    """Caso límite: with_individual_plot=True también funciona."""
    res = plot_features_cat_regression(df_cat_reg, "precio", with_individual_plot=True)
    assert isinstance(res, list)
    plt.close('all')


def test_plot_cat_caso_error(df_cat_reg):
    """Caso de error: df inválido -> None."""
    assert plot_features_cat_regression("no es df", "precio") is None
    plt.close('all')


# ----------------------------- detect_outliers -----------------------------




# ----------------------------- Dataset numérico para pruebas de clasificación -----------------------------

@pytest.fixture
def df_num_cat():
    """Dataset numérico para CLASIFICACIÓN: 'y' depende de 'x_buena'; 'x_ruido' es irrelevante."""
    return pd.DataFrame({
        "y": ["a", "a", "b", "b", "c", "c"],   # target categórico
        "x_buena": [2, 4, 6, 8, 10, 12],       # correlación perfecta con y
        "x_ruido": [5, 1, 4, 2, 6, 3],         # sin relación
    })


# ----------------------------- get_features_num_classification -----------------------------

def test_get_features_num_classification_devuelve_lista(df_num_cat):
    """
    Caso correcto: input válido → retorna una lista.
    """
    resultado = get_features_num_classification(df_num_cat, target="y")
    assert isinstance(resultado, list)


def test_get_features_num_classification_seleccion_correcta(df_num_cat):
    """
    'x_buena' debe ser seleccionada; 'x_ruido' debe ser descartada.
    """
    resultado = get_features_num_classification(df_num_cat, target="y")

    assert "x_buena" in resultado
    assert "x_ruido" not in resultado
    assert resultado == ["x_buena"]  # lista exacta esperada


def test_get_features_num_classification_sin_numericas():
    """
    Si el DataFrame no tiene columnas numéricas → retorna lista vacía.
    """
    df = pd.DataFrame({
        "y": ["a", "b", "a"],
        "color": ["rojo", "azul", "rojo"]
    })

    resultado = get_features_num_classification(df, target="y")
    assert resultado == []


# ----------------------------- Dataset categórico para pruebas de clasificación -----------------------------

@pytest.fixture
def df_cat_cat():
    """Dataset categórico para CLASIFICACIÓN: 'y' depende de 'clase'; 'color' es irrelevante."""
    return pd.DataFrame({
        "y": pd.Series(
            ["bajo", "bajo", "bajo", "alto", "alto", "alto", "medio", "medio", "medio"],
            dtype="object"
        ),
        "clase": pd.Series(
            ["baja", "baja", "baja", "alta", "alta", "alta", "media", "media", "media"],
            dtype="object"
        ),
        "color": pd.Series(
            ["rojo", "azul", "rojo", "azul", "rojo", "azul", "rojo", "azul", "rojo"],
            dtype="object"
        ),
    })


# ----------------------------- get_features_cat_classification -----------------------------

def test_get_features_cat_classification_devuelve_lista(df_cat_cat):
    """
    Caso correcto: input válido → retorna una lista.
    """
    resultado = get_features_cat_classification(df_cat_cat, target="y")
    assert isinstance(resultado, list)


def test_get_features_cat_classification_seleccion_correcta(df_cat_cat):
    """
    'clase' debe ser seleccionada; 'color' debe ser descartada.
    """
    resultado = get_features_cat_classification(df_cat_cat, target="y")

    assert "clase" in resultado
    assert "color" not in resultado
    assert resultado == ["clase"]  # lista exacta esperada


def test_get_features_cat_classification_sin_categoricas():
    """
    Si el DataFrame no tiene columnas categóricas → retorna lista vacía.
    """
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "precio": [10, 20, 30]
    })

    resultado = get_features_cat_classification(df, target="y")
    assert resultado == []
