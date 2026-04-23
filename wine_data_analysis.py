
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from ucimlrepo import fetch_ucirepo

# Loading the dataset"
wine_quality = fetch_ucirepo(name="Wine Quality")
wine_df = wine_quality.data.original

'''
Preprocesamiento de Datos:
'''
print("\n\n============ Dataset head ============\n")
print(wine_df.head()) # veiwing the top 5 rows of the dataset

print("\n\n============ Dataset Info ============\n")
print(wine_df.info()) # checking the info of the dataset
#varibale objetivo
print("\n\n============ Variable Objetivo ============\n")
print(wine_df.quality.value_counts()) # checking the distribution of the target variable
plt.figure(figsize = (12,8))
sns.countplot(x = wine_df['quality'].astype(object))
plt.suptitle('Quality Score Value Counts')
plt.tight_layout()
plt.show()

print("\n\n============ Dataset Summary ============\n")
print(wine_df.describe()) # checking the summary statistics of the dataset

#Pre-processing dataset
print("\nNumber of Duplicates in dataset:",wine_df.duplicated().sum())
wine_df = wine_df.drop_duplicates()
print(wine_df.head())
#Eliminamos la variables binarias/no continuas para usar
wine_df = wine_df.drop(columns=['color'])
print(wine_df.info())

'''
Reducción de Dimensionalidad usando PCA:
'''
#Normalización de variables numéricas
cont_features = wine_df[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']] #Excluimos la variable objetivo 'quality' y la columna 'Id'
classes = wine_df['quality']

features_scaled = StandardScaler().fit_transform(cont_features) # normalizing the features# checking the shape of the scaled features and original dataset

pca = decomposition.PCA().fit(features_scaled)
features_pca = pca.transform(features_scaled)#Projecting the data to the new PCA space

print("\n\n============ PCA Transformed Dataset Analysis ============\n")  
print("Eigenvalues:", pca.explained_variance_)
print("\nExplained variance",pca.explained_variance_ratio_.cumsum()*100)

#Project the data in a space of reduced dimensionality:
#Nos quedamos con el número de componentes necesarios para explicar el 95% de la varianza cumulativa
for i in range(len(pca.explained_variance_ratio_)):
    if pca.explained_variance_ratio_.cumsum()[i]*100 > 95:
        n_components = i+1
        break

print(f"Number of PCA components to keep: {n_components}")    
wine_df_pca = pd.DataFrame(features_pca[:,0:n_components],columns=[f'PC{j+1}' for j in range(n_components)])

print("\n\n============ PCA Transformed Dataset Head ============\n")
print('Dimensionalidad datos en espacio PCA reducido = {}'.format(wine_df_pca.shape))
print(wine_df_pca.head()) # checking the first 5 rows of the new PCA dataset

sns.pairplot(wine_df_pca)
plt.show()

'''
Clasificación:
'''
np.random.seed(42)
#Crear lo sets de train y test
X_train, X_test, y_train, y_test = train_test_split(wine_df_pca, classes, test_size=0.3, random_state=42)#Proporción 70/30 para train y test
print("\n\n============ Train and Test Set Shapes ============\n")
print(f"Train set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")


#Entrenar los modelos clasificadores
#Primer modelo: LDA (Análisis Discriminante Lineal)
n_classes = len(np.unique(y_train))
n = min(n_classes - 1, n_components)
lda = LinearDiscriminantAnalysis(n_components=n) #n_components = Number of components (<= min(n_classes - 1, n_features))
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

#Segundo modelo: NAÏVE BAYES
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

'''
Evaluación de Modelos:
'''
print("\n\n============ LDA ============\n")
print("Accuracy LDA:", accuracy_score(y_test, y_pred_lda))
mlda = confusion_matrix(y_test, y_pred_lda)
#print(classification_report(y_test, y_pred_lda))


print("\n\n============ NAÏVE BAYES ============\n")
print("Accuracy Naive Bayes:", accuracy_score(y_test, y_pred_gnb))
mbayes = confusion_matrix(y_test, y_pred_gnb)
#print(classification_report(y_test, y_pred_gnb))

fig, ax = plt.subplots(1, 2, figsize=(16, 4),sharey=True)

sns.heatmap(mlda,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[0])
ax[0].set_title('Matriz de Confusión - LDA')
ax[0].set_xlabel('Etiqueta Predicha')
ax[0].set_ylabel('Etiqueta Real')

sns.heatmap(mbayes,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[1])
ax[1].set_title('Matriz de Confusión - Naïve Bayes')
ax[1].set_xlabel('Etiqueta Predicha')
ax[1].set_ylabel('Etiqueta Real')

plt.tight_layout()
plt.show()

print(y_test.value_counts())