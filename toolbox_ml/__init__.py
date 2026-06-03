"""
Toolbox ML consiste en paquete de herramientas automatizadas para Análisis Exploratorio de Datos (EDA)
y selección de características orientadas a modelos de regresión.
Archivo destinado a definir la versión de la librería y para importar las funciones principales directamente a la raíz, haciendo un import directamente.
"""

# Definimos la versión del paquete
__version__ = "0.1.0"

# Importamos las funciones
from .eda.core import (
    describe_df,
    tipifica_variables,
    # get_features_num_regression,
    # plot_features_num_regression,
    # get_features_cat_regression,
    # plot_features_cat_regression
)

# Definimoas qué funciones se exportan públicamente
__all__ = [
    "describe_df",
    "tipifica_variables",
    "get_features_num_regression",
    "plot_features_num_regression",
    "get_features_cat_regression",
    "plot_features_cat_regression"
]

# Con este archivo llamaremos a las funciones en nuestro notebook demostración.