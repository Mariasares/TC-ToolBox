import pandas as pd
import numpy as np
from typing import List, Optional

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen detallado de un DataFrame de pandas.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame que se va a analizar.
    
    Retorna:
    pd.DataFrame: Un DataFrame con tipos, % nulos, valores únicos y cardinalidad por columna.
    """
    # TODO: Validar que df sea un pd.DataFrame
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
    # TODO: Validar tipos de entrada y umbrales
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
    pass

def plot_features_num_regression(df: pd.DataFrame, target_col: str, umbral_corr: float = 0.5, columns: List[str] = None) -> Optional[List[str]]:
    """
    Grafica las variables numéricas correlacionadas con el target mediante pairplots o scatterplots.
    
    Nota: Debe llamar internamente a get_features_num_regression. Max 5 columnas por gráfico.
    """
    pass

def get_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> Optional[List[str]]:
    """
    Identifica variables categóricas estadísticamente significativas respecto a un target numérico
    utilizando tests de hipótesis (T-Test/Mann-Whitney o ANOVA/Kruskal-Wallis).
    """
    pass

def plot_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05, columns: List[str] = None) -> Optional[List[str]]:
    """
    Grafica diagramas de caja (boxplots) para las variables categóricas significativas respecto al target.
    """
    pass