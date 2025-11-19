import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        # Cargar los datos
        df = pd.read_csv('data.csv')
        
        # Preprocesamiento de datos
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
        print("ANÁLISIS 1: PREDICCIÓN DE TEMPERATURA MÁXIMA")
        
        features_temp = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                        'Month', 'DayOfYear', 'weather_encoded', 'cloud_encoded']
        
        X_temp = df[features_temp]
        y_temp = df['maxtemp']
        
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_temp.fit(X_train_temp, y_train_temp)
        y_pred_temp = rf_temp.predict(X_test_temp)
        
        mse_temp = mean_squared_error(y_test_temp, y_pred_temp)
        r2_temp = r2_score(y_test_temp, y_pred_temp)
        
        print(f"R² Temperatura: {r2_temp:.4f}")
        print(f"RMSE Temperatura: {np.sqrt(mse_temp):.4f}")
        
        # ANÁLISIS 2: Clasificación del tipo de clima
        print("\nANÁLISIS 2: CLASIFICACIÓN DEL TIPO DE CLIMA")
        
        features_weather = ['maxtemp', 'mintemp', 'pressure', 'humidity', 
                          'mean wind speed', 'Month', 'cloud_encoded']
        
        X_weather = df[features_weather]
        y_weather = df['weather_encoded']
        
        X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(
            X_weather, y_weather, test_size=0.2, random_state=42, stratify=y_weather
        )
        
        rf_weather = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_weather.fit(X_train_weather, y_train_weather)
        y_pred_weather = rf_weather.predict(X_test_weather)
        
        accuracy_weather = accuracy_score(y_test_weather, y_pred_weather)
        print(f"Precisión en clasificación de clima: {accuracy_weather:.4f}")
        
        # Crear visualizaciones
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Importancia características temperatura
        feature_importance_temp = pd.DataFrame({
            'feature': features_temp,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.subplot(2, 2, 1)
        sns.barplot(data=feature_importance_temp, x='importance', y='feature')
        plt.title('Importancia de Características - Temperatura')
        
        # Gráfico 2: Predicciones vs reales temperatura
        plt.subplot(2, 2, 2)
        plt.scatter(y_test_temp, y_pred_temp, alpha=0.6)
        plt.plot([y_test_temp.min(), y_test_temp.max()], 
                [y_test_temp.min(), y_test_temp.max()], 'r--')
        plt.xlabel('Real')
        plt.ylabel('Predicho')
        plt.title(f'Temperatura (R² = {r2_temp:.3f})')
        
        # Gráfico 3: Temperatura por mes
        plt.subplot(2, 2, 3)
        df.groupby('Month')['maxtemp'].mean().plot(kind='bar')
        plt.title('Temperatura Máxima Promedio por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Temperatura (°C)')
        
        # Gráfico 4: Distribución de tipos de clima
        plt.subplot(2, 2, 4)
        df['weather'].value_counts().plot(kind='bar')
        plt.title('Distribución de Tipos de Clima')
        plt.xlabel('Tipo de Clima')
        plt.ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig('weather_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar resultados en archivo
        with open('model_results.txt', 'w') as f:
            f.write("RESULTADOS DEL MODELO RANDOM FOREST\n")
            f.write("=" * 50 + "\n")
            f.write(f"Predicción Temperatura - R²: {r2_temp:.4f}\n")
            f.write(f"Clasificación Clima - Precisión: {accuracy_weather:.4f}\n")
            f.write(f"Característica más importante temperatura: {feature_importance_temp.iloc[0]['feature']}\n")
        
        print("Análisis completado exitosamente!")
        print("Archivos generados: weather_analysis_results.png, model_results.txt")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
