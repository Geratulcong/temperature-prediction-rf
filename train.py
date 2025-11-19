import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('data.csv')

# Exploración inicial de los datos
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Preprocesamiento de datos
# Convertir fecha a datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

# Extraer características temporales
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear

# Limpiar nombres de columnas (eliminar espacios)
df.columns = df.columns.str.strip()

# Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Preparar datos para Random Forest
# Codificar variables categóricas
le_weather = LabelEncoder()
le_cloud = LabelEncoder()

df['weather_encoded'] = le_weather.fit_transform(df['weather'])
df['cloud_encoded'] = le_cloud.fit_transform(df['cloud'])

# ANÁLISIS 1: Predicción de temperatura máxima
print("\n" + "="*50)
print("ANÁLISIS 1: PREDICCIÓN DE TEMPERATURA MÁXIMA")
print("="*50)

# Seleccionar características para predicción de temperatura
features_temp = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                'Month', 'DayOfYear', 'weather_encoded', 'cloud_encoded']

X_temp = df[features_temp]
y_temp = df['maxtemp']

# Dividir datos
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

# Entrenar Random Forest para regresión
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X_train_temp, y_train_temp)

# Predicciones
y_pred_temp = rf_temp.predict(X_test_temp)

# Evaluación
mse_temp = mean_squared_error(y_test_temp, y_pred_temp)
r2_temp = r2_score(y_test_temp, y_pred_temp)

print(f"MSE Temperatura: {mse_temp:.4f}")
print(f"R² Temperatura: {r2_temp:.4f}")
print(f"RMSE Temperatura: {np.sqrt(mse_temp):.4f}")

# Importancia de características
feature_importance_temp = pd.DataFrame({
    'feature': features_temp,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportancia de características para temperatura:")
print(feature_importance_temp)

# ANÁLISIS 2: Clasificación del tipo de clima
print("\n" + "="*50)
print("ANÁLISIS 2: CLASIFICACIÓN DEL TIPO DE CLIMA")
print("="*50)

# Seleccionar características para clasificación
features_weather = ['maxtemp', 'mintemp', 'pressure', 'humidity', 
                   'mean wind speed', 'Month', 'cloud_encoded']

X_weather = df[features_weather]
y_weather = df['weather_encoded']

# Dividir datos
X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(
    X_weather, y_weather, test_size=0.2, random_state=42, stratify=y_weather
)

# Entrenar Random Forest para clasificación
rf_weather = RandomForestClassifier(n_estimators=100, random_state=42)
rf_weather.fit(X_train_weather, y_train_weather)

# Predicciones
y_pred_weather = rf_weather.predict(X_test_weather)

# Evaluación
accuracy_weather = accuracy_score(y_test_weather, y_pred_weather)

print(f"Precisión en clasificación de clima: {accuracy_weather:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test_weather, y_pred_weather, 
                          target_names=le_weather.classes_))

# Importancia de características para clasificación
feature_importance_weather = pd.DataFrame({
    'feature': features_weather,
    'importance': rf_weather.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportancia de características para clasificación de clima:")
print(feature_importance_weather)

# ANÁLISIS 3: Predicción de humedad
print("\n" + "="*50)
print("ANÁLISIS 3: PREDICCIÓN DE HUMEDAD")
print("="*50)

features_humidity = ['maxtemp', 'mintemp', 'pressure', 'mean wind speed', 
                    'Month', 'weather_encoded', 'cloud_encoded']

X_humidity = df[features_humidity]
y_humidity = df['humidity']

# Dividir datos
X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(
    X_humidity, y_humidity, test_size=0.2, random_state=42
)

# Entrenar Random Forest para humedad
rf_humidity = RandomForestRegressor(n_estimators=100, random_state=42)
rf_humidity.fit(X_train_hum, y_train_hum)

# Predicciones
y_pred_hum = rf_humidity.predict(X_test_hum)

# Evaluación
mse_hum = mean_squared_error(y_test_hum, y_pred_hum)
r2_hum = r2_score(y_test_hum, y_pred_hum)

print(f"MSE Humedad: {mse_hum:.4f}")
print(f"R² Humedad: {r2_hum:.4f}")
print(f"RMSE Humedad: {np.sqrt(mse_hum):.4f}")

# VISUALIZACIONES
plt.figure(figsize=(15, 10))

# Gráfico 1: Importancia características temperatura
plt.subplot(2, 2, 1)
sns.barplot(data=feature_importance_temp, x='importance', y='feature')
plt.title('Importancia de Características - Temperatura Máxima')
plt.xlabel('Importancia')

# Gráfico 2: Predicciones vs Valores reales temperatura
plt.subplot(2, 2, 2)
plt.scatter(y_test_temp, y_pred_temp, alpha=0.6)
plt.plot([y_test_temp.min(), y_test_temp.max()], [y_test_temp.min(), y_test_temp.max()], 'r--')
plt.xlabel('Temperatura Real (°C)')
plt.ylabel('Temperatura Predicha (°C)')
plt.title(f'Predicción de Temperatura (R² = {r2_temp:.3f})')

# Gráfico 3: Importancia características clasificación clima
plt.subplot(2, 2, 3)
sns.barplot(data=feature_importance_weather, x='importance', y='feature')
plt.title('Importancia de Características - Clasificación de Clima')
plt.xlabel('Importancia')

# Gráfico 4: Predicciones vs Valores reales humedad
plt.subplot(2, 2, 4)
plt.scatter(y_test_hum, y_pred_hum, alpha=0.6)
plt.plot([y_test_hum.min(), y_test_hum.max()], [y_test_hum.min(), y_test_hum.max()], 'r--')
plt.xlabel('Humedad Real (%)')
plt.ylabel('Humedad Predicha (%)')
plt.title(f'Predicción de Humedad (R² = {r2_hum:.3f})')

plt.tight_layout()
plt.show()

# Análisis temporal de temperaturas
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
df.groupby('Month')['maxtemp'].mean().plot(kind='bar')
plt.title('Temperatura Máxima Promedio por Mes')
plt.xlabel('Mes')
plt.ylabel('Temperatura Máxima (°C)')

plt.subplot(1, 2, 2)
df.groupby('Month')['mintemp'].mean().plot(kind='bar', color='orange')
plt.title('Temperatura Mínima Promedio por Mes')
plt.xlabel('Mes')
plt.ylabel('Temperatura Mínima (°C)')

plt.tight_layout()
plt.show()

# Resumen de resultados
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)
print(f"Predicción Temperatura Máxima - R²: {r2_temp:.4f}")
print(f"Clasificación Tipo de Clima - Precisión: {accuracy_weather:.4f}")
print(f"Predicción Humedad - R²: {r2_hum:.4f}")

# Características más importantes en general
print("\nCARACTERÍSTICAS MÁS IMPORTANTES:")
print("1. Para temperatura:", feature_importance_temp.iloc[0]['feature'])
print("2. Para clasificación de clima:", feature_importance_weather.iloc[0]['feature'])

# Ejemplo de predicción para un nuevo día
print("\n" + "="*50)
print("EJEMPLO DE PREDICCIÓN")
print("="*50)

# Crear un ejemplo de predicción
sample_data = {
    'mintemp': [25.0],
    'pressure': [755.0],
    'humidity': [80.0],
    'mean wind speed': [2.5],
    'Month': [6],
    'DayOfYear': [150],
    'weather_encoded': [le_weather.transform(['Haze'])[0]],
    'cloud_encoded': [le_cloud.transform(['No Significant Clouds'])[0]]
}

sample_df = pd.DataFrame(sample_data)
predicted_temp = rf_temp.predict(sample_df)[0]
print(f"Temperatura máxima predicha: {predicted_temp:.1f}°C")
