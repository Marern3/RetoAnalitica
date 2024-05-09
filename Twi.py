import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("twitter_dataset.csv")

# Separar para el Diagrama de cajas y bigotes
df_Tweet = df['Tweet_ID']
df_Retweets = df['Retweets']
df_Likes = df['Likes']

# Mostrar las primeras filas del dataframe
print(df.head())

# Resumen estadístico
print(df.describe())

# Excluir las variables no numéricas antes de calcular la correlación
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Diagramas de cajas y bigotes
#Tweets
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_Tweet)
plt.title('Diagrama de Cajas y Bigotes')
plt.xticks(rotation=45)
plt.xlabel('Tweets')
plt.show()
# Retweets
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_Retweets)
plt.title('Diagrama de Cajas y Bigotes')
plt.xticks(rotation=45)
plt.xlabel('Retweets')
plt.show()
# Likes
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_Likes)
plt.title('Diagrama de Cajas y Bigotes')
plt.xticks(rotation=45)
plt.xlabel('Likes')
plt.show()

# Histogramas
plt.figure(figsize=(10, 6))
numeric_df.hist(bins=20, figsize=(10,6))
plt.suptitle('Histogramas')
plt.show()

# Mapa de calor de la correlación
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='PuBu')
plt.title('Mapa de Calor de Correlación')
plt.show()