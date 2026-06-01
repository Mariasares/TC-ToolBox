# Para que el paquete sea instalable y Pytest funcione correctamente
from setuptools import setup, find_packages

setup(
    name="toolbox_ml",
    version="0.1.0",
    packages=find_packages(),
    install_packages=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
    ],
)