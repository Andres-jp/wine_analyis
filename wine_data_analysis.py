
'''
Objetivo:
    El objetivo de este ejercicio es aplicar técnicas de reducción de dimensionalidad y clasificadores probabilísticos para analizar un conjunto de datos, comparar los rendimientos de diferentes clasificadores y determinar cuál es más efectivo para predecir la calidad del vino en función de sus características.

Contexto:
    El conjunto de datos "wine quality dataset" contiene información relacionada con variantes rojas y blancas del vino portugués "Vinho Verde". Los atributos incluyen la acidez, el azúcar residual, el alcohol, y otros, con el objetivo de predecir la calidad del vino.

Tareas:

Preprocesamiento de Datos:

    Cargar el dataset "wine quality dataset".
    Realizar un análisis exploratorio inicial para entender las características de los datos, incluyendo estadísticas descriptivas y visualizaciones de las distribuciones de las características y la variable objetivo.
    Limpiar y preparar los datos para el análisis, lo que puede incluir la gestión de valores faltantes y la normalización de variables numéricas.

Reducción de Dimensionalidad usando PCA:

    Aplicar PCA (Análisis de Componentes Principales) para reducir la dimensionalidad del conjunto de datos.
    Seleccionar el número de componentes principales necesarios para capturar una cantidad significativa de la variabilidad en los datos.
    Transformar los datos a este nuevo espacio de características reducido.

Clasificación:

    Utilizar dos clasificadores probabilísticos para modelar y predecir la variable objetivo.
    Entrenar los modelos usando los datos transformados por PCA.

Evaluación de Modelos:

    Evaluar el rendimiento de cada modelo clasificador utilizando métricas como la precisión, la matriz de confusión, ...
    Comparar los rendimientos de los dos clasificadores basándose en estas métricas.
'''
#data analysis of wine quiality data set
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Loading the dataset"
current_path = os.getcwd()
ruta_completa = os.path.join(current_path, 'WineQT.csv')
wine_df = pd.read_csv(ruta_completa)

'''
Preprocesamiento de Datos:
'''
print("============ Dataset head ============")
print(wine_df.head()) # veiwing the top 5 rows of the dataset
print("============ Dataset Info ============")
print(wine_df.info()) # checking the info of the dataset
#varibale objetivo
print("============ Variable Objetivo ============")
print(wine_df.quality.value_counts()) # checking the distribution of the target variable
plt.figure(figsize = (12,8))
sns.countplot(x = wine_df['quality'].astype(object))
plt.suptitle('Quality Score Value Counts')
plt.tight_layout()
plt.show()
print("============ Dataset Summary ============")
print(wine_df.describe()) # checking the summary statistics of the dataset
#Normalización de variables numéricas
from sklearn.preprocessing import StandardScaler


#Excluimos la variable objetivo 'quality' y la columna 'Id'
columnas_a_escalar = wine_df.drop(columns=['quality', 'Id']).columns

wine_df_scaled = pd.DataFrame(StandardScaler().fit_transform(wine_df[columnas_a_escalar]), columns=columnas_a_escalar)


#checking the normalized data
print("============ Normalized Dataset Summary ============")
print(wine_df_scaled.describe())

'''
Reducción de Dimensionalidad usando PCA:
'''
from sklearn import decomposition
pca = decomposition.PCA(n_components=9).fit(wine_df_scaled)
print("============ PCA Explained Variance Ratio ============")
print(100*pca.explained_variance_ratio_) # checking the explained variance ratio of the PCA
print("============ PCA Cumulative Explained Variance ============")
print(100*pca.explained_variance_ratio_.cumsum()) # checking the cumulative explained variance of the PCA

