import pandas as pd
import pytest
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