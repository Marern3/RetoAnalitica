import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("twitter_dataset.csv")

# Eliminar variables
df.drop(columns=['Tweet_ID'], inplace=True)

# Identificar las variables no numéricas
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
print("Variables no numéricas:")
print(non_numeric_cols)

# Eliminar las variables no numéricas antes de ajustar el modelo K-Means
numeric_df = df.select_dtypes(include=[np.number])
X = numeric_df.values

# Determinar un valor de k utilizando el método del codo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia')
plt.title('Método del Codo')
plt.grid(True)
plt.show()

# Utilizar k=3 para el algoritmo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Centros de los clusters
centers = kmeans.cluster_centers_
print("Centros del algoritmo K-Means:")
print(centers)

# Calcular la distancia entre los centros
distances = pd.DataFrame(distance_matrix(centers, centers), index=range(1, centers.shape[0] + 1), columns=range(1, centers.shape[0] + 1))
print("Distancia entre los centros de los clusters:")
print(distances)

# Visualización de los centros en relación con los datos
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('Centros de los Clusters')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.show()