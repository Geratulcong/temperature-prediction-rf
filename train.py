import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import datetime

# Generar datos sintéticos de temperatura (o cargar dataset real)
def generate_temperature_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    data = {
        'date': dates,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'year': dates.year,
        'humidity': np.random.normal(60, 15, len(dates)),
        'pressure': np.random.normal(1013, 10, len(dates)),
        'wind_speed': np.random.gamma(2, 2, len(dates))
    }
    
    # Temperatura base con estacionalidad
    base_temp = 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    # Añadir efectos de otras variables y ruido
    temperature = (base_temp + 
                  0.1 * data['humidity'] + 
                  0.05 * data['pressure'] + 
                  0.2 * data['wind_speed'] + 
                  np.random.normal(0, 2, len(dates)))
    
    data['temperature'] = temperature
    return pd.DataFrame(data)

# Cargar/generar datos
print("📊 Generando datos de temperatura...")
df = generate_temperature_data()

# Guardar dataset generado
df.to_csv('temperature_data.csv', index=False)

# Preparar características y target
features = ['day_of_year', 'month', 'year', 'humidity', 'pressure', 'wind_speed']
X = df[features]
y = df['temperature']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
print("🌲 Entrenando modelo Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📈 Métricas del modelo:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Guardar modelo
joblib.dump(model, 'temperature_model.pkl')
print("💾 Modelo guardado como 'temperature_model.pkl'")

# Visualización
plt.figure(figsize=(12, 6))

# Gráfico 1: Predicciones vs Real
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Temperatura Real (°C)')
plt.ylabel('Temperatura Predicha (°C)')
plt.title(f'Random Forest - R² = {r2:.2f}')

# Gráfico 2: Importancia de características
plt.subplot(1, 2, 2)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importancia')
plt.title('Importancia de Características')

plt.tight_layout()
plt.savefig('temperature_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Entrenamiento completado!")
