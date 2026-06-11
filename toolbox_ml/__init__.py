import os
import sys

# Para que el programa ubique la carpeta raíz del proyecto
# Puente para las rutas absolutas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Toolbox ML consiste en paquete de herramientas automatizadas para Análisis Exploratorio de Datos (EDA)
y selección de características orientadas a modelos de regresión.
Archivo destinado a definir la versión de la librería y para importar las funciones principales directamente a la raíz, haciendo un import directo.
"""

# Definimos la versión del paquete
__version__ = "1.0.0"

# Importamos las funciones de las desarrolladoras desde core
from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression,
    detect_outliers,  # <-- Nueva función Bonus 1
)

# Definimoas qué funciones se exportan públicamente
__all__ = [
    "describe_df",
    "tipifica_variables",
    "get_features_num_regression",
    "plot_features_num_regression",
    "get_features_cat_regression",
    "plot_features_cat_regression",
    "detect_outliers" # <-- Nueva
]

# Con este archivo llamaremos a las funciones en nuestro notebook demostración.