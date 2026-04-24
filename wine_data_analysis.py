
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

### Inicialización

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_validate
from ucimlrepo import fetch_ucirepo

# Loading the dataset"
wine_quality = fetch_ucirepo(name="Wine Quality")
wine_df = wine_quality.data.original

### Preprocesamiento de Datos

print("\n\n============ Raw Dataset head ============\n")
print(wine_df.head()) 

print("\n\n============ Raw Dataset Info ============\n")
print(wine_df.info())

#Limpiar y preparar los datos para el análisis
print("\nNumber of Duplicates in dataset:",wine_df.duplicated().sum())
wine_df = wine_df.drop_duplicates().reset_index(drop=True)
#Eliminamos la variables binarias/no continuas para usar
wine_df = wine_df.drop(columns=['color'])

print("\n\n============ Pre-processing dataset Info ============\n")
print(wine_df.info())

#varibale objetivo
print("\n\n============ Variable Objetivo ============\n")
print(wine_df.quality.value_counts()) # checking distribution
#Graficamos distribución variable objetivo 'quality'
plt.figure(figsize = (12,8))
sns.countplot(x = wine_df['quality'].astype(object))
plt.suptitle('Quality Score Value Counts')
plt.tight_layout()
plt.show()
sns.pairplot(wine_df)
plt.show()

print("\n\n============ Dataset Summary ============\n")
print(wine_df.describe()) # resumen estadísitcas del dataset


### Reducción de Dimensionalidad usando PCA

#Normalización de variables numéricas
cont_features = wine_df[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']] #Excluimos la variable objetivo 'quality' y la columna 'Id'
classes = wine_df['quality']
n_features = cont_features.shape[1]
print(n_features)

features_scaled = StandardScaler().fit_transform(cont_features) # normalizing the features# checking the shape of the scaled features and original dataset

pca = decomposition.PCA().fit(features_scaled)
features_pca = pca.transform(features_scaled)#Projecting the data to the new PCA space

print("\n\n============ PCA Transformed Dataset Analysis ============\n")  
print("Eigenvalues:", pca.explained_variance_)
print("\nExplained variance",pca.explained_variance_ratio_.cumsum()*100)

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



### Clasificación

np.random.seed(42)
#Crear lo sets de train y test
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(wine_df_pca, classes, test_size=0.3, random_state=42)#Proporción 70/30 para train y test
print("\n\n============ Train and Test Set Shapes ============\n")
print(f"Train set shape: {X_train_pca.shape}, {y_train_pca.shape}")
print(f"Test set shape: {X_test_pca.shape}, {y_test_pca.shape}")

X_train, X_test, y_train, y_test = train_test_split(features_scaled, classes, test_size=0.3, random_state=42)#Proporción 70/30 para train y test (datos escalados)
print("\n\n============ Train and Test Set Shapes (sin PCA, escalados) ============\n")
print(f"Train set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")


#Entrenar los modelos clasificadores
#Primer modelo: LDA (Análisis Discriminante Lineal) CON PCA
n_classes = len(np.unique(y_train))
n_lda_components = min(n_classes - 1, n_components)
lda_pca = LinearDiscriminantAnalysis(n_components=n_lda_components) #n_components = Number of components (<= min(n_classes - 1, n_features))
lda_pca.fit(X_train_pca, y_train_pca)
y_pred_lda_pca = lda_pca.predict(X_test_pca)

#Primer modelo: LDA (Análisis Discriminante Lineal) SIN PCA
n_classes = len(np.unique(y_train))
n_lda_components_nopca = min(n_classes - 1, n_features)
lda = LinearDiscriminantAnalysis(n_components=n_lda_components_nopca) #n_components basado en features originales, no PCA
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

#Segundo modelo: NAÏVE BAYES CON PCA
gnb_pca = GaussianNB()
gnb_pca.fit(X_train_pca, y_train_pca)
y_pred_gnb_pca = gnb_pca.predict(X_test_pca)

#Segundo modelo: NAÏVE BAYES SIN PCA
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)


### Evaluación de Modelos

print("\n\n============ LDA CON PCA ============\n")
print("Accuracy LDA:", accuracy_score(y_test_pca, y_pred_lda_pca))
mlda_pca = confusion_matrix(y_test_pca, y_pred_lda_pca)

print("\n\n============ LDA SIN PCA ============\n")
print("Accuracy LDA:", accuracy_score(y_test, y_pred_lda))
mlda = confusion_matrix(y_test, y_pred_lda)

print("\n\n============ NAÏVE BAYES CON PCA ============\n")
print("Accuracy Naive Bayes:", accuracy_score(y_test_pca, y_pred_gnb_pca))
mbayes_pca = confusion_matrix(y_test_pca, y_pred_gnb_pca)

print("\n\n============ NAÏVE BAYES SIN PCA ============\n")
print("Accuracy Naive Bayes:", accuracy_score(y_test, y_pred_gnb))
mbayes = confusion_matrix(y_test, y_pred_gnb)

fig, ax = plt.subplots(2, 2, figsize=(16, 8),sharey=True)
ax = ax.flatten()

sns.heatmap(mlda,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[0])
ax[0].set_title('Matriz de Confusión - LDA')
ax[0].set_xlabel('Etiqueta Predicha')
ax[0].set_ylabel('Etiqueta Real')

sns.heatmap(mlda_pca,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[1])
ax[1].set_title('Matriz de Confusión - LDA PCA')
ax[1].set_xlabel('Etiqueta Predicha')
ax[1].set_ylabel('Etiqueta Real')

sns.heatmap(mbayes,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[2])
ax[2].set_title('Matriz de Confusión - Naïve Bayes')
ax[2].set_xlabel('Etiqueta Predicha')
ax[2].set_ylabel('Etiqueta Real')

sns.heatmap(mbayes_pca,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[3])
ax[3].set_title('Matriz de Confusión - Naïve Bayes PCA')
ax[3].set_xlabel('Etiqueta Predicha')
ax[3].set_ylabel('Etiqueta Real')

scores_metrics = ['precision_macro', 'recall_macro']

print("\n\n============ Cross Validation LDA SIN PCA ============\n")
cv_results = cross_validate(lda, X_train, y_train, cv=5, scoring=scores_metrics)
print("Precision Macro: %0.2f (+/- %0.2f)" % (cv_results['test_precision_macro'].mean(), cv_results['test_precision_macro'].std() * 2))
print("Recall Macro: %0.2f (+/- %0.2f)" % (cv_results['test_recall_macro'].mean(), cv_results['test_recall_macro'].std() * 2))

print("\n\n============ Cross Validation LDA CON PCA ============\n")
cv_results = cross_validate(lda_pca, X_train_pca, y_train_pca, cv=5, scoring=scores_metrics)
print("Precision Macro: %0.2f (+/- %0.2f)" % (cv_results['test_precision_macro'].mean(), cv_results['test_precision_macro'].std() * 2))
print("Recall Macro: %0.2f (+/- %0.2f)" % (cv_results['test_recall_macro'].mean(), cv_results['test_recall_macro'].std() * 2))

print("\n\n============ Cross Validation Naive Bayes SIN PCA ============\n")
cv_results = cross_validate(gnb, X_train, y_train, cv=5, scoring=scores_metrics)
print("Precision Macro: %0.2f (+/- %0.2f)" % (cv_results['test_precision_macro'].mean(), cv_results['test_precision_macro'].std() * 2))
print("Recall Macro: %0.2f (+/- %0.2f)" % (cv_results['test_recall_macro'].mean(), cv_results['test_recall_macro'].std() * 2))

print("\n\n============ Cross Validation Naive Bayes CON PCA ============\n")
cv_results = cross_validate(gnb_pca, X_train_pca, y_train_pca, cv=5, scoring=scores_metrics)
print("Precision Macro: %0.2f (+/- %0.2f)" % (cv_results['test_precision_macro'].mean(), cv_results['test_precision_macro'].std() * 2))
print("Recall Macro: %0.2f (+/- %0.2f)" % (cv_results['test_recall_macro'].mean(), cv_results['test_recall_macro'].std() * 2))

plt.tight_layout()
plt.show()


### Usando SMOTE

x_dataset = wine_df.drop(columns=['quality'])
y_dataset = wine_df['quality']

scaler_smote = StandardScaler()
feature_names = cont_features.columns.tolist() # reutilizamos las columnas ya definidas
scaled_features_smote = scaler_smote.fit_transform(x_dataset[feature_names])

from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=4)

final_features, final_labels = oversample.fit_resample(scaled_features_smote, y_dataset)


final_features_df = pd.DataFrame(final_features, columns=feature_names)
final_labels_df = pd.DataFrame(final_labels, columns=['quality'])
wine_SMOTE_df = pd.concat([final_features_df, final_labels_df], axis=1)


#varibale objetivo
print("\n\n============ Variable Objetivo tras SMOTE ============\n")
print(wine_SMOTE_df.quality.value_counts()) # checking distribution
#Graficamos distribución variable objetivo 'quality'
plt.figure(figsize = (12,8))
sns.countplot(x = wine_SMOTE_df['quality'].astype(object))
plt.suptitle('Quality Score Value Counts')
plt.tight_layout()
plt.show()


### PCA
pca_smote = decomposition.PCA().fit(final_features_df)
features_pca_smote = pca_smote.transform(final_features_df)#Projecting the data to the new PCA space

print("\n\n============ PCA Transformed Dataset Analysis ============\n")  
print("Eigenvalues:", pca_smote.explained_variance_)
print("\nExplained variance",pca_smote.explained_variance_ratio_.cumsum()*100)

#Nos quedamos con el número de componentes necesarios para explicar el 95% de la varianza cumulativa
n_components_smote = n_features  # valor por defecto si no se alcanza el 95%
for i in range(len(pca_smote.explained_variance_ratio_)):
    if pca_smote.explained_variance_ratio_.cumsum()[i]*100 > 95:
        n_components_smote = i+1
        break

print(f"Number of PCA components to keep: {n_components_smote}")    
wine_df_pca_smote = pd.DataFrame(features_pca_smote[:,0:n_components_smote],columns=[f'PC{j+1}' for j in range(n_components_smote)])

print("\n\n============ PCA Transformed Dataset Head ============\n")
print('Dimensionalidad datos en espacio PCA reducido = {}'.format(wine_df_pca_smote.shape))
print(wine_df_pca_smote.head()) # checking the first 5 rows of the new PCA dataset


### Clasificadores

np.random.seed(42)
#Crear lo sets de train y test
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(wine_df_pca_smote, final_labels_df['quality'].values.ravel(), test_size=0.3, random_state=42)#Proporción 70/30 para train y test
print("\n\n============ Train and Test Set Shapes ============\n")
print(f"Train set shape: {X_train_2.shape}, {y_train_2.shape}")
print(f"Test set shape: {X_test_2.shape}, {y_test_2.shape}")

#Entrenar los modelos clasificadores
#Primer modelo: LDA 
n_classes_smote = len(np.unique(y_train_2))
n_lda_smote = min(n_classes_smote - 1, n_components_smote)
lda_smote = LinearDiscriminantAnalysis(n_components=n_lda_smote) #n_components = Number of components (<= min(n_classes - 1, n_features))
lda_smote.fit(X_train_2, y_train_2)
y_pred_lda_smote = lda_smote.predict(X_test_2)

#Segundo modelo: NAÏVE BAYES
gnb_smote = GaussianNB()
gnb_smote.fit(X_train_2, y_train_2)
y_pred_gnb_smote = gnb_smote.predict(X_test_2)


### Evaluación

mbayes_smote = confusion_matrix(y_test_2, y_pred_gnb_smote)
mlda_smote = confusion_matrix(y_test_2, y_pred_lda_smote)

fig, ax = plt.subplots(1, 2, figsize=(16, 4),sharey=True)

sns.heatmap(mlda_smote,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[0])
ax[0].set_title('Matriz de Confusión - LDA (SMOTE + PCA)')
ax[0].set_xlabel('Etiqueta Predicha')
ax[0].set_ylabel('Etiqueta Real')

sns.heatmap(mbayes_smote,annot = True,linewidths=0.5,linecolor="green",fmt = ".0f",ax=ax[1])
ax[1].set_title('Matriz de Confusión - Naïve Bayes (SMOTE + PCA)')
ax[1].set_xlabel('Etiqueta Predicha')
ax[1].set_ylabel('Etiqueta Real')

plt.tight_layout()
plt.show()

print("\n\n============ Cross Validation LDA (SMOTE + PCA) ============\n")
print("Accuracy LDA:", accuracy_score(y_test_2, y_pred_lda_smote))
cv_results = cross_validate(lda_smote, X_train_2, y_train_2, cv=5, scoring=scores_metrics)
print("\nPrecision Macro: %0.2f (+/- %0.2f)" % (cv_results['test_precision_macro'].mean(), cv_results['test_precision_macro'].std() * 2))
print("Recall Macro: %0.2f (+/- %0.2f)" % (cv_results['test_recall_macro'].mean(), cv_results['test_recall_macro'].std() * 2))

print("\n\n============ Cross Validation Naive Bayes (SMOTE + PCA) ============\n")
print("Accuracy Naive Bayes:", accuracy_score(y_test_2, y_pred_gnb_smote))
cv_results = cross_validate(gnb_smote, X_train_2, y_train_2, cv=5, scoring=scores_metrics)
print("\nPrecision Macro: %0.2f (+/- %0.2f)" % (cv_results['test_precision_macro'].mean(), cv_results['test_precision_macro'].std() * 2))
print("Recall Macro: %0.2f (+/- %0.2f)" % (cv_results['test_recall_macro'].mean(), cv_results['test_recall_macro'].std() * 2))


