"""
Módulo: toolbox_ML.py
Descripción: Colección de funciones auxiliares para análisis exploratorio de datos (EDA) y Machine Learning.
Equipo / Grupo: M004
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def describe_df(df):
    """
    Genera un resumen descriptivo de un DataFrame de pandas.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame que se quiere analizar.
    
    Retorna:
    pd.DataFrame: Un nuevo DataFrame con tipos, % nulos, valores únicos y cardinalidad por columna.
    """
    # TODO: Implementar por ????
    pass

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las variables de un DataFrame en tipos sugeridos según su cardinalidad.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame a analizar.
    umbral_categoria (int): Límite de cardinalidad para considerar una variable como categoría.
    umbral_continua (int): Límite de cardinalidad para considerar una variable como numérica continua.
    
    Retorna:
    pd.DataFrame: Un DataFrame con los nombres de las columnas y su tipo sugerido.
    """
    # TODO: Implementar por ???
    pass

# Puedes agregar aquí los nombres/esqueletos de las 4 funciones restantes siguiendo esta estructura...
