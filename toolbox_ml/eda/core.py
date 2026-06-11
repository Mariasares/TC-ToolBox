
import pandas as pd
import numpy as np
from typing import List, Optional
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, mannwhitneyu, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------- describe_df -----------------------------

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.
    
    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas: 'tipo', 'porcentaje_nulos', 'valores_unicos',
        'porcentaje_cardinalidad'.
        Retorna None si el input no es un DataFrame válido.
    """
    
    # Comprobación de que si es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el objeto proporcionado no es un DataFrame.")
        return None

    # Crear el DataFrame resultado
    resultado = pd.DataFrame(index=df.columns)

    # Tipo de dato
    resultado["tipo"] = df.dtypes.astype(str)

    # Porcentaje de nulos
    resultado["porcentaje_nulos"] = (df.isna().mean() * 100).round(2)

    # Valores únicos
    resultado["valores_unicos"] = df.nunique()

    # Porcentaje de cardinalidad
    resultado["porcentaje_cardinalidad"] = ((df.nunique() / len(df)) * 100).round(2)


    return resultado


# ----------------------------- tipifica_variables -----------------------------


def tipifica_variables(df: pd.DataFrame, umbral_categorica: int, umbral_continua: float) -> pd.DataFrame:
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de cardinalidad.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categorica: Umbral para categorizar las variables categóricas
        umbral_continua: Float

    Retorna:
        pd.DataFrame: DataFrame con dos columnas: 'nombre_variable', 'tipo_sugerido'
        Retorna None si el input no es un DataFrame válido.
        Retorna None si el umbral_categorica no es un tipo válido (int).
        Retorna None si el umbral_continua no es un tipo válido (float)
    """
    
    # Comprobación de DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el objeto proporcionado no es un DataFrame.")
        return None
    
    # Comprobación umbral_categorica, debe ser entero positivo
    if not isinstance(umbral_categorica, int) or umbral_categorica <= 0:
        print("Error: umbral_categorica debe ser un entero positivo.")
        return None

    # comprobación umbral_continua, debe ser float entre 0 y 100
    if not isinstance(umbral_continua, float) or not (0 <= umbral_continua <= 100):
        print("Error: umbral_continua debe ser un float entre 0 y 100.")
        return None
    
    
    # Cardinalidad y porcentaje_cardinalidad
    cardinalidad = df.nunique()
    porcentaje_cardinalidad = df.nunique() / len(df) * 100

    # Clasificación variables
    tipos = []

    for col in df.columns:
        card = cardinalidad[col]
        pct = porcentaje_cardinalidad[col]

        if card == 2:
            tipo = "Binaria"

        elif card < umbral_categorica:
            tipo = "Categórica"

        elif card >= umbral_categorica and pct >= umbral_continua:
            tipo = "Numérica Continua"

        else:
            tipo = "Numérica Discreta"

        tipos.append(tipo)
    
    # DF resultado
    resultado = pd.DataFrame({
        "nombre_variable": df.columns,
        "tipo_sugerido": tipos
    })

    return resultado



# ----------------------------------------------------------------------------- #
#  Helper privado: comprobaciones de entrada comunes a las funciones numéricas.
#  Lo separamos para no repetir el mismo bloque de "ifs" en dos funciones.
#  Devuelve True si todo está bien, o False (e imprime el motivo) si algo falla.
# ----------------------------------------------------------------------------- #
def _validar_num(df, target_col, umbral_corr, pvalue) -> bool:
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pandas DataFrame.")
        return False
    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el DataFrame.")
        return False
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: 'target_col' ('{target_col}') debe ser una columna numérica.")
        return False
    # bool es subclase de int en Python, por eso lo excluimos explícitamente
    if isinstance(umbral_corr, bool) or not isinstance(umbral_corr, (int, float)) \
            or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return False
    if pvalue is not None:
        if isinstance(pvalue, bool) or not isinstance(pvalue, (int, float)) \
                or not (0 <= pvalue <= 1):
            print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
            return False
    return True


# ----------------------------- get_features_num_regression -----------------------------


def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None
) -> list:
    """
    Selecciona las columnas numéricas que correlacionan con el target.

    Argumentos:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): nombre de la columna objetivo (debe ser numérica).
        umbral_corr (float): umbral de correlación absoluta, entre 0 y 1.
        pvalue (float, opcional): si se indica (entre 0 y 1), aplica un filtro
            adicional de significación estadística (p-valor < pvalue).

    Retorna:
        list: nombres de las columnas numéricas cuya correlación de Pearson
        con 'target_col', en valor absoluto, supera 'umbral_corr' (y, si se
        pide, son estadísticamente significativas).
        Retorna None si alguna comprobación de entrada falla.
    """
    # --- comprobaciones de entrada (si fallan -> None) ---
    if not _validar_num(df, target_col, umbral_corr, pvalue):
        return None

    # --- variables que defino YO para trabajar ---
    resultado = []                                          # aquí guardo los nombres válidos
    columnas_numericas = df.select_dtypes(include=np.number).columns  # las descubre solas

    # --- recorro cada columna numérica ---
    for col in columnas_numericas:
        if col == target_col:
            continue  # el target no se compara consigo mismo

        # me quedo solo con las filas sin NaN en ambas columnas
        datos = df[[col, target_col]].dropna()
        # pearsonr necesita al menos 2 datos y que la columna no sea constante
        if len(datos) < 2 or datos[col].nunique() < 2:
            continue

        # pearsonr devuelve DOS valores: el coeficiente (r) y el p-valor (p)
        r, p = pearsonr(datos[col], datos[target_col])

        # filtro 1: la correlación absoluta supera el umbral
        if abs(r) > umbral_corr:
            # filtro 2 (opcional): la correlación es significativa
            if pvalue is None or p < pvalue:
                resultado.append(col)

    return resultado


# ----------------------------- plot_features_num_regression -----------------------------


def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
) -> list:
    """
    Pinta un pairplot del target frente a las columnas numéricas que cumplen
    los criterios de correlación (misma lógica que get_features_num_regression).

    Si 'columns' está vacía, usa todas las columnas numéricas como candidatas.
    Si hay que representar muchas columnas, las reparte en grupos de máximo
    5 columnas (incluyendo siempre el target) y pinta un pairplot por grupo.

    Argumentos:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): columna objetivo (numérica).
        columns (list): columnas candidatas. Si está vacía, se usan todas las numéricas.
        umbral_corr (float): umbral de correlación absoluta, entre 0 y 1.
        pvalue (float, opcional): filtro de significación estadística.

    Retorna:
        list: columnas que finalmente cumplen los criterios y se han pintado.
        Retorna None si alguna comprobación de entrada falla.
    """
    # mismas comprobaciones que la función anterior
    if not _validar_num(df, target_col, umbral_corr, pvalue):
        return None

    # si no me dan columnas, uso todas las numéricas (quitando el target)
    if not columns:
        columns = list(df.select_dtypes(include=np.number).columns)
        if target_col in columns:
            columns.remove(target_col)

    # me quedo solo con candidatas que existan y no sean el target
    candidatas = [c for c in columns if c in df.columns and c != target_col]

    # reutilizo la lógica de selección sobre un sub-DataFrame (target + candidatas)
    sub = df[[target_col] + candidatas]
    seleccionadas = get_features_num_regression(sub, target_col, umbral_corr, pvalue)
    if seleccionadas is None:
        return None
    if len(seleccionadas) == 0:
        print("Ninguna columna supera los criterios: no hay nada que pintar.")
        return seleccionadas

    # pinto en grupos de máximo 4 features + el target = 5 columnas por pairplot
    # (si tu profe prefiere 5 features, cambia este 4 por un 5)
    max_features_por_grupo = 4
    for i in range(0, len(seleccionadas), max_features_por_grupo):
        grupo = seleccionadas[i:i + max_features_por_grupo]
        sns.pairplot(df[[target_col] + grupo].dropna())
        plt.suptitle(f"Pairplot: {target_col} vs {grupo}", y=1.02)
        plt.show()

    return seleccionadas


# ----------------------------------------------------------------------------- #
#  Helper privado: comprobaciones comunes a las funciones categóricas.
# ----------------------------------------------------------------------------- #

def _validar_cat(df, target_col, pvalue) -> bool:
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pandas DataFrame.")
        return False
    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el DataFrame.")
        return False
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: 'target_col' ('{target_col}') debe ser una columna numérica.")
        return False
    if isinstance(pvalue, bool) or not isinstance(pvalue, (int, float)) \
            or not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser un número entre 0 y 1.")
        return False
    return True


# ----------------------------- get_features_cat_regression -----------------------------


def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Selecciona las columnas categóricas con relación estadística significativa
    con el target numérico. Elige el test automáticamente según la cardinalidad:
        - 2 categorías   -> Mann-Whitney U (compara distribuciones de 2 grupos)
        - >2 categorías  -> ANOVA de un factor (compara medias de varios grupos)

    Argumentos:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): columna objetivo (numérica).
        pvalue (float): nivel de significación, entre 0 y 1 (por defecto 0.05).

    Retorna:
        list: nombres de las columnas categóricas con p-valor < pvalue.
        Retorna None si alguna comprobación de entrada falla.
    """
    if not _validar_cat(df, target_col, pvalue):
        return None

    resultado = []
    # considero categóricas las de tipo texto, categoría o booleano
    columnas_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in columnas_categoricas:
        # construyo un grupo de valores del target por cada categoría (sin NaN)
        grupos = [
            df.loc[df[col] == cat, target_col].dropna()
            for cat in df[col].dropna().unique()
        ]
        # descarto grupos vacíos
        grupos = [g for g in grupos if len(g) > 0]
        n_categorias = len(grupos)

        if n_categorias < 2:
            continue  # con menos de 2 grupos no hay test posible

        try:
            if n_categorias == 2:
                # exactamente 2 categorías -> Mann-Whitney U
                stat, p = mannwhitneyu(grupos[0], grupos[1])
            else:
                # más de 2 categorías -> ANOVA de un factor
                stat, p = f_oneway(*grupos)  # el * "desempaqueta" la lista de grupos
        except Exception:
            continue  # si el test falla (datos degenerados), salto la columna

        if p < pvalue:
            resultado.append(col)

    return resultado


# ----------------------------- plot_features_cat_regression -----------------------------


def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list:
    """
    Para cada columna categórica que supera el test (misma lógica que
    get_features_cat_regression), pinta histogramas del target agrupados
    por cada valor de la categoría.

    Si 'columns' está vacía, usa todas las categóricas como candidatas.
        - with_individual_plot=False (por defecto): todo en una figura con subplots.
        - with_individual_plot=True: una figura independiente por variable.

    Retorna:
        list: columnas categóricas que han superado el test y se han pintado.
        Retorna None si alguna comprobación de entrada falla.
    """
    if not _validar_cat(df, target_col, pvalue):
        return None

    if not columns:
        columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    candidatas = [c for c in columns if c in df.columns and c != target_col]

    sub = df[[target_col] + candidatas]
    seleccionadas = get_features_cat_regression(sub, target_col, pvalue)
    if seleccionadas is None:
        return None
    if len(seleccionadas) == 0:
        print("Ninguna columna categórica supera el test: no hay nada que pintar.")
        return seleccionadas

    if with_individual_plot:
        # una figura por variable
        for col in seleccionadas:
            plt.figure(figsize=(7, 4))
            sns.histplot(data=df, x=target_col, hue=col, element="step")
            plt.title(f"Distribución de {target_col} por {col}")
            plt.show()
    else:
        # todas las variables en una sola figura con subplots
        n = len(seleccionadas)
        fig, axes = plt.subplots(n, 1, figsize=(7, 4 * n))
        if n == 1:
            axes = [axes]  # con un solo subplot, axes no es una lista
        for ax, col in zip(axes, seleccionadas):
            sns.histplot(data=df, x=target_col, hue=col, element="step", ax=ax)
            ax.set_title(f"Distribución de {target_col} por {col}")
        plt.tight_layout()
        plt.show()

    return seleccionadas


# ----------------------------- detect_outliers -----------------------------


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = None) -> dict:
    """
    Detecta valores atípicos (outliers) en las columnas numéricas de un DataFrame.
    
    Argumentos:
        df (pd.DataFrame): El DataFrame que contiene los datos a analizar.
        method (str): El método de detección; puede ser 'iqr' (Rango Intercuartílico) 
                      o 'z_score' (Puntuación Z). Por defecto es 'iqr'.
        threshold (float, opcional): El umbral para la detección. 
                                     Si es 'iqr', por defecto multiplica por 1.5.
                                     Si es 'z_score', por defecto busca valores fuera de +/- 3.0.
                                     
    Retorna:
        dict: Un diccionario donde cada clave es el nombre de la columna numérica y el valor 
              es otro diccionario con: 'count' (cantidad), 'percentage' (%) e 'indices' (lista de índices).
    """
    # 1. Comprobaciones de entrada correctas
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' debe ser un DataFrame.")
        return {}
        
    method = method.lower()
    if method not in ['iqr', 'z_score']:
        print("El método debe ser 'iqr' o 'z_score'.")
        return {}
        
    # Para seleccionar automáticamente solo las columnas numéricas
    columns_num = df.select_dtypes(include=[np.number]).columns
    
    # El diccionario donde guardaremos los resultados finales
    outliers_summary = {}
    
    # 2. Bucle principal para analizar columna por columna
    for col in columns_num:
        # Ignoramos columnas que estén completamente vacías
        if df[col].dropna().empty:
            continue
            
        # Creamos una serie limpia sin nulos para calcular los límites estadísticos correctamente
        col_clean = df[col].dropna()
        outlier_mask = pd.Series(False, index=df.index)
        
        # MÉTODO 1: RANGO INTERCUARTÍLICO (IQR)
        if method == 'iqr':
            k = threshold if threshold is not None else 1.5
            q1 = col_clean.quantile(0.25)
            q3 = col_clean.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (k * iqr)
            upper_bound = q3 + (k * iqr)
            
            # Identificamos qué índices del DataFrame original caen fuera de los límites
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        # MÉTODO 2: Z-SCORE (PUNTUACIÓN Z)
        elif method == 'z_score':
            limit = threshold if threshold is not None else 3.0
            mean = col_clean.mean()
            std = col_clean.std()
            
            # Si la desviación estándar es 0 (columna constante), no hay outliers posibles
            if std == 0:
                continue
                
            # Calculamos el Z-score absoluto para cada valor
            z_scores = (df[col] - mean) / std
            outlier_mask = z_scores.abs() > limit
            
        # 3. Recopilación de resultados de la columna actual
        # Extraemos los índices reales del DataFrame donde se activa la máscara de outlier
        detected_indices = df.index[outlier_mask].tolist()
        count = len(detected_indices)
        percentage = round((count / len(df)) * 100, 2)
        
        # Guardamos en el diccionario estructurado
        outliers_summary[col] = {
            'count': count,
            'percentage': percentage,
            'indices': detected_indices
        }
        
    return outliers_summary


# ----------------------------- get_features_num_classification -----------------------------


def get_features_num_classification(df, target, pvalue_rango=0.05):
    """
    Selección de features numéricas relevantes para un modelo de 
    clasificación usando el método estadístico ANOVA.

    Parámetros:
        - df : DataFrame
        - target : str - Nombre de la columna objetivo.
        - pvalue_rango : float - Umbral de significancia (default=0.05).

    Retorna:
    --------
    features_num_sel : Lista de nombres de features numéricas seleccionadas.
    """
    # Separar X e y
    y = df[target]
    X = df.drop(columns=[target])

    # Seleccionar solo columnas numéricas
    features_num = X.select_dtypes(include=["int64", "float64"])

    # Si no hay columnas numéricas retorna una lista vacía
    if features_num.shape[1] == 0:
        return []

    # Creamos una lista vacía para que se guarden las features seleccionadas
    features_num_sel = []

    # Para cada columna numérica aplicamos el método y calculamos p_value
    for col in features_num.columns:

        # Crear lista de grupos: una lista por cada clase del target
        grupos = [df[df[target] == clase][col].dropna() 
                  for clase in df[target].unique()]

        # Método ANOVA
        f_val, p_val = stats.f_oneway(*grupos)

        # Seleccionar si p < umbral, se rechaza la hipótesis nula (la variable es dependiente)
        if p_val < pvalue_rango:
            features_num_sel.append(col)

    return features_num_sel


# ----------------------------- get_features_cat_classification -----------------------------


def get_features_cat_classification(df, target, pvalue_rango=0.05):
    """
    Selección de features categórcias relevantes para un modelo de 
    clasificación usando el método estadístico Chi-cuadrado.

    Parámetros:
        - df : DataFrame
        - target : str - Nombre de la columna objetivo.
        - pvalue_rango : float - Umbral de significancia (default=0.05).

    Retorna:
    --------
    features_cat_sel : Lista de nombres de features numéricas seleccionadas.
    """
    # Separar X e y
    y = df[target]
    X = df.drop(columns=[target])
    # Convertir columnas string (StringDtype) a object para evitar errores con SciPy
    X = X.astype({col: "object" for col in X.select_dtypes(include="string").columns})

    # Seleccionar solo columnas categóricas
    features_cat = X.select_dtypes(include=["object", "category"])

    # Si no hay columnas numéricas retorna una lista vacía
    if features_cat.shape[1] == 0:
        return []

    # Creamos una lista vacía para que se guarden las features seleccionadas
    features_cat_sel = []

    # Para cada columna cat aplicamos el método y calculamos p_value
    for col in features_cat.columns:

        # Crear tabla de contingencia
        tabla_contingencia = pd.crosstab(df[col], df[target])

        # Test Chi-cuadrado
        chi2, p_val, dof, expected = chi2_contingency(tabla_contingencia)

        # Seleccionar si p < umbral, se rechaza la hipótesis nula (la variable es dependiente)
        if p_val < pvalue_rango:
            features_cat_sel.append(col)

    return features_cat_sel



