import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import List, Optional

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen detallado de un DataFrame de pandas.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame que se va a analizar.
    
    Retorna:
    pd.DataFrame: Un DataFrame con tipos, % nulos, valores únicos y cardinalidad por columna.
    """
    # TODO: Desarrollador 1, Marta
    pass

def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float) -> pd.DataFrame:
    """
    Sugiere el tipo de variable (Binaria, Categórica, Numérica Continua o Numérica Discreta)
    basándose en su cardinalidad y porcentaje de cardinalidad.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame a analizar.
    umbral_categoria (int): Límite de valores únicos para considerar una variable como categórica.
    umbral_continua (float): Porcentaje de cardinalidad mínimo para considerar una variable como continua.
    
    Retorna:
    pd.DataFrame: Un DataFrame con las columnas 'Variable' y 'Tipo_Sugerido'.
    """
    # TODO: Desarrollador 1, Marta
    pass

def get_features_num_regression(df: pd.DataFrame, target_col: str, umbral_corr: float = 0.5) -> Optional[List[str]]:
    """
    Devuelve las columnas numéricas cuya correlación con el target supera un umbral absoluto.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame de datos.
    target_col (str): El nombre de la columna objetivo (numérica).
    umbral_corr (float): Umbral de correlación de Pearson (entre 0 y 1).
    
    Retorna:
    List[str] o None: Lista de features que superan el umbral, o None si hay errores.
    """
    # TODO: Desarrollador 2, Claudia
    pass

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0.0,
    pvalue: Optional[float] = None
) -> Optional[List[str]]:
    """
    Pinta un pairplot de target_col junto con las columnas que cumplan los criterios 
    de correlación. Si supera 5 columnas, divide en subplots de máximo 5.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Columna objetivo numérica.
        columns (list): Lista de columnas candidatas (vacía = todas las numéricas).
        umbral_corr (float): Umbral de correlación en valor absoluto.
        pvalue (float, opcional): Nivel de significación estadística.

    Retorna:
        List[str]: Lista de columnas finalmente representadas.
        Retorna None si falla alguna comprobación de entrada.
    """
    # TODO: Desarrollador 2, Claudia
    pass

def get_features_cat_regression(
    df: pd.DataFrame, 
    target_col: str, 
    pvalue: float = 0.05
) -> Optional[List[str]]:
    """
    Devuelve las columnas categóricas cuya relación con target_col sea significativa 
    usando Mann-Whitney U (2 categorías) o ANOVA (>2 categorías).

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Columna objetivo numérica.
        pvalue (float): Nivel de significación estadística (defecto 0.05).

    Retorna:
        List[str]: Lista de columnas categóricas estadísticamente significativas.
        Retorna None si falla alguna comprobación de entrada.
    """
    # TODO: Desarrollador 2/3 , Claudia y yo??
    pass

def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> Optional[List[str]]:
    """
    Pinta histogramas agrupados de target_col por cada variable categórica seleccionada.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Columna objetivo numérica.
        columns (list): Lista de columnas candidatas (vacía = todas las categóricas).
        pvalue (float): Nivel de significación estadística.
        with_individual_plot (bool): False para un único plot con subplots, True para independientes.

    Retorna:
        List[str]: Lista de columnas categóricas finalmente representadas.
        Retorna None si falla alguna comprobación de entrada.
    """
    # TODO: Desarrollador 2/3, Claudia y yo??
    pass