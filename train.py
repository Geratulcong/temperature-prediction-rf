import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

def main():
    try:
        # Cargar los datos
        print("Cargando datos desde data.csv...")
        df = pd.read_csv('data.csv')
        
        # Preprocesamiento de datos
        print("Preprocesando datos...")
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Codificar variables categóricas
        le_weather = LabelEncoder()
        le_cloud = LabelEncoder()
        
        df['weather_encoded'] = le_weather.fit_transform(df['weather'])
        df['cloud_encoded'] = le_cloud.fit_transform(df['cloud'].astype(str))
        
        # ANÁLISIS 1: Predicción de temperatura máxima
        print("\n" + "="*50)
        print("ANÁLISIS 1: PREDICCIÓN DE TEMPERATURA MÁXIMA")
        print("="*50)
        
        features_temp = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                        'Month', 'DayOfYear', 'weather_encoded', 'cloud_encoded']
        
        X_temp = df[features_temp]
        y_temp = df['maxtemp']
        
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        print("Entrenando modelo de Random Forest para temperatura...")
        rf_temp.fit(X_train_temp, y_train_temp)
        y_pred_temp = rf_temp.predict(X_test_temp)
        
        mse_temp = mean_squared_error(y_test_temp, y_pred_temp)
        r2_temp = r2_score(y_test_temp, y_pred_temp)
        
        print(f"✓ R² Temperatura: {r2_temp:.4f}")
        print(f"✓ RMSE Temperatura: {np.sqrt(mse_temp):.4f}")
        
        # ANÁLISIS 2: Clasificación del tipo de clima
        print("\n" + "="*50)
        print("ANÁLISIS 2: CLASIFICACIÓN DEL TIPO DE CLIMA")
        print("="*50)
        
        features_weather = ['maxtemp', 'mintemp', 'pressure', 'humidity', 
                          'mean wind speed', 'Month', 'cloud_encoded']
        
        X_weather = df[features_weather]
        y_weather = df['weather_encoded']
        
        # SIN ESTRATIFICACIÓN
        X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(
            X_weather, y_weather, test_size=0.2, random_state=42
        )
        
        rf_weather = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Entrenando modelo de Random Forest para clasificación de clima...")
        rf_weather.fit(X_train_weather, y_train_weather)
        y_pred_weather = rf_weather.predict(X_test_weather)
        
        accuracy_weather = accuracy_score(y_test_weather, y_pred_weather)
        print(f"✓ Precisión en clasificación de clima: {accuracy_weather:.4f}")
        
        # ANÁLISIS 3: Predicción de humedad
        print("\n" + "="*50)
        print("ANÁLISIS 3: PREDICCIÓN DE HUMEDAD")
        print("="*50)
        
        features_humidity = ['maxtemp', 'mintemp', 'pressure', 'mean wind speed', 
                           'Month', 'weather_encoded', 'cloud_encoded']
        
        X_humidity = df[features_humidity]
        y_humidity = df['humidity']
        
        X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(
            X_humidity, y_humidity, test_size=0.2, random_state=42
        )
        
        rf_humidity = RandomForestRegressor(n_estimators=100, random_state=42)
        print("Entrenando modelo de Random Forest para humedad...")
        rf_humidity.fit(X_train_hum, y_train_hum)
        y_pred_hum = rf_humidity.predict(X_test_hum)
        
        mse_hum = mean_squared_error(y_test_hum, y_pred_hum)
        r2_hum = r2_score(y_test_hum, y_pred_hum)
        
        print(f"✓ R² Humedad: {r2_hum:.4f}")
        print(f"✓ RMSE Humedad: {np.sqrt(mse_hum):.4f}")
        
        # Crear visualizaciones
        print("\nGenerando visualizaciones...")
        plt.figure(figsize=(15, 12))
        
        # Gráfico 1: Importancia características temperatura
        feature_importance_temp = pd.DataFrame({
            'feature': features_temp,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.subplot(2, 3, 1)
        sns.barplot(data=feature_importance_temp, x='importance', y='feature')
        plt.title('Importancia de Características - Temperatura')
        plt.xlabel('Importancia')
        
        # Gráfico 2: Predicciones vs reales temperatura
        plt.subplot(2, 3, 2)
        plt.scatter(y_test_temp, y_pred_temp, alpha=0.6)
        plt.plot([y_test_temp.min(), y_test_temp.max()], 
                [y_test_temp.min(), y_test_temp.max()], 'r--', linewidth=2)
        plt.xlabel('Temperatura Real (°C)')
        plt.ylabel('Temperatura Predicha (°C)')
        plt.title(f'Predicción de Temperatura (R² = {r2_temp:.3f})')
        
        # Gráfico 3: Importancia características clasificación
        feature_importance_weather = pd.DataFrame({
            'feature': features_weather,
            'importance': rf_weather.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.subplot(2, 3, 3)
        sns.barplot(data=feature_importance_weather, x='importance', y='feature')
        plt.title('Importancia de Características - Clima')
        plt.xlabel('Importancia')
        
        # Gráfico 4: Predicciones vs reales humedad
        plt.subplot(2, 3, 4)
        plt.scatter(y_test_hum, y_pred_hum, alpha=0.6, color='green')
        plt.plot([y_test_hum.min(), y_test_hum.max()], 
                [y_test_hum.min(), y_test_hum.max()], 'r--', linewidth=2)
        plt.xlabel('Humedad Real (%)')
        plt.ylabel('Humedad Predicha (%)')
        plt.title(f'Predicción de Humedad (R² = {r2_hum:.3f})')
        
        # Gráfico 5: Temperatura por mes
        plt.subplot(2, 3, 5)
        df.groupby('Month')['maxtemp'].mean().plot(kind='bar', color='skyblue')
        plt.title('Temperatura Máxima Promedio por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Temperatura (°C)')
        
        # Gráfico 6: Distribución de tipos de clima
        plt.subplot(2, 3, 6)
        df['weather'].value_counts().plot(kind='bar', color='lightcoral')
        plt.title('Distribución de Tipos de Clima')
        plt.xlabel('Tipo de Clima')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('weather_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar resultados en archivo
        with open('model_results.txt', 'w') as f:
            f.write("RESULTADOS DEL MODELO RANDOM FOREST\n")
            f.write("=" * 50 + "\n")
            f.write(f"Predicción Temperatura - R²: {r2_temp:.4f}\n")
            f.write(f"Clasificación Clima - Precisión: {accuracy_weather:.4f}\n")
            f.write(f"Predicción Humedad - R²: {r2_hum:.4f}\n")
            f.write(f"Característica más importante temperatura: {feature_importance_temp.iloc[0]['feature']}\n")
            f.write(f"Característica más importante clima: {feature_importance_weather.iloc[0]['feature']}\n")
        
        # Guardar resultados en JSON
        results = {
            "temperature_r2": float(r2_temp),
            "temperature_rmse": float(np.sqrt(mse_temp)),
            "weather_accuracy": float(accuracy_weather),
            "humidity_r2": float(r2_hum),
            "humidity_rmse": float(np.sqrt(mse_hum)),
            "top_temp_feature": feature_importance_temp.iloc[0]['feature'],
            "top_weather_feature": feature_importance_weather.iloc[0]['feature']
        }
        
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print("="*50)
        print(f"✓ Predicción Temperatura - R²: {r2_temp:.4f}")
        print(f"✓ Clasificación Clima - Precisión: {accuracy_weather:.4f}")
        print(f"✓ Predicción Humedad - R²: {r2_hum:.4f}")
        print(f"✓ Archivos generados:")
        print(f"  - weather_analysis_results.png")
        print(f"  - model_results.txt")
        print(f"  - results.json")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
