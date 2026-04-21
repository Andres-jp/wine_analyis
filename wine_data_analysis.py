
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

'''
Reducción de Dimensionalidad usando PCA:
'''
#Escalado de las características numéricas
features = wine_df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] #Excluimos la variable objetivo 'quality' y la columna 'Id'
classes = wine_df['quality']

features_scaled = StandardScaler().fit_transform(features) # normalizing the features# checking the shape of the scaled features and original dataset

from sklearn import decomposition
pca = decomposition.PCA().fit(features_scaled)
features_pca = pca.transform(features_scaled)#Projecting the data to the new PCA space

print("============ PCA Transformed Dataset Analysis ============")  
print("Eigenvalues:", pca.explained_variance_)
print("Explained variance",pca.explained_variance_ratio_.cumsum()*100)

#Project the data in a space of reduced dimensionality:
#Nos quedamos con el número de componentes necesarios para explicar el 95% de la varianza cumulativa
for i in range(len(pca.explained_variance_ratio_)):
    if pca.explained_variance_ratio_.cumsum()[i]*100 > 95:
        n_components = i+1
        break
print(f"Number of PCA components to keep: {n_components}")    
wine_df_pca = pd.DataFrame(features_pca[:,0:n_components],columns=[f'PC{j+1}' for j in range(n_components)])

print("============ PCA Transformed Dataset Head ============")
print('Dimensionalidad datos en espacio PCA reducido = {}'.format(wine_df_pca.shape))
print(wine_df_pca.head()) # checking the first 5 rows of the new PCA dataset

sns.pairplot(wine_df_pca)
plt.show()

'''
Clasificación:
'''
