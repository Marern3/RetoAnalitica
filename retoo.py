import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

# Cargar los datos
df = pd.read_csv("covid19_tweets.csv")

# Verificar la cantidad de datos y las variables que contiene
print("Cantidad de datos:", len(df))
print("Variables en el conjunto de datos:", list(df.columns))
print("Tipo de variables:")
print(df.dtypes)

# Análisis descriptivo de las variables numéricas
print("\nAnálisis descriptivo:")
print(df.describe())

# Identificar variables numéricas
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

df['date'] = pd.to_datetime(df['date'])  # Convertir la columna de fecha a tipo datetime
df['date'].hist(bins=30)
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Tweets')
plt.title('Cantidad de Tweets por Fecha')
plt.show()

# Diagramas de cajas y bigotes
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_columns])
plt.title("Diagrama de cajas y bigotes")
plt.xticks(rotation=45)
plt.show()

# Histogramas
plt.figure(figsize=(12, 6))
df[numeric_columns].hist(bins=20)
plt.suptitle("Histogramas", y=1.02)
plt.show()

# Mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de calor de correlación")
plt.show()

# Eliminar variables que no aportan información relevante
df_cleaned = df.drop(['user_name', 'user_location', 'user_description', 'user_created', 'hashtags', 'source', 'is_retweet'], axis=1)

# Escalar las variables numéricas
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned.select_dtypes(include=['int64', 'float64']))

# Determinar un rango de valores de k
k_range = range(1, 15)
inertia = []

# Aplicar k-means para diferentes valores de k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Seleccionar el valor de k
k = 4

# Aplicar k-means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df_scaled)
df_cleaned['cluster'] = kmeans.labels_

# Visualización de resultados
sns.scatterplot(data=df_cleaned, x='user_followers', y='user_friends', hue='cluster', palette='Set1')
plt.xlabel('Seguidores del Usuario')
plt.ylabel('Amigos del Usuario')
plt.title('Clustering de Usuarios por Seguidores y Amigos')
plt.show()

# Centros de los clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers, columns=df_cleaned.select_dtypes(include=['int64', 'float64']).columns)
print("\nCentros de los clusters:")
print(centers_df)

# Visualización de los centros en relación con los datos
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_scaled)
closest_points = df_cleaned.iloc[closest]

plt.figure(figsize=(12, 6))
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=df_cleaned['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.scatter(centers_df.iloc[:, 0], centers_df.iloc[:, 1], c='red', s=200, marker='o')
plt.scatter(closest_points.iloc[:, 0], closest_points.iloc[:, 1], c='blue', s=100, marker='x')
plt.title('Visualización de los centros en relación con los datos')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.show()

# Calcular la distancia entre los centros
distance_matrix = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        distance_matrix[i, j] = np.linalg.norm(cluster_centers[i] - cluster_centers[j])

print("\nDistancia entre los centros de los clusters:")
print(pd.DataFrame(distance_matrix, columns=["Cluster " + str(i+1) for i in range(k)], index=["Cluster " + str(i+1) for i in range(k)]))
