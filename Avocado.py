import pandas as pd

# Cargar los datos
df = pd.read_csv("avocado_full.csv")

# Filtrar los datos para la regiones
df_NewYork = df[df['region'] == 'NewYork']
df_California = df[df['region'] == 'California']
df_Charlotte = df[df['region'] == 'Charlotte']

# Seleccionar las columnas específicas
df_NewYork = df_NewYork[['region', 'Date', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']]
# Promedio
promedio_total_bags_NY = df_NewYork['Total Bags'].mean()
print("Promedio de Total Bags New York:", promedio_total_bags_NY)

# Desviación estándar
desviacion_total_bags_NY = df_NewYork['Total Bags'].std()
print("Desviación estándar de Total Bags New York:", desviacion_total_bags_NY)


df_California = df_California[['region', 'Date', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']]
# Promedio
promedio_total_bags_Cal = df_California['Total Bags'].mean()
print("Promedio de Total Bags California:", promedio_total_bags_Cal)

# Desviación estándar
desviacion_total_bags_Cal = df_California['Total Bags'].std()
print("Desviación estándar de Total Bags California:", desviacion_total_bags_Cal)


df_Charlotte = df_Charlotte[['region', 'Date', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']]
# Promedio
promedio_total_bags_Char = df_Charlotte['Total Bags'].mean()
print("Promedio de Total Bags Charlotte:", promedio_total_bags_Char)

# Desviación estándar
desviacion_total_bags_Char = df_Charlotte['Total Bags'].std()
print("Desviación estándar de Total Bags Charlotte:", desviacion_total_bags_Char)

# Mostrar los datos filtrados
print(df_NewYork)
print(df_California)
print(df_Charlotte)
