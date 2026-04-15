import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el dataset Housing.csv desde la misma carpeta que el script
df = pd.read_csv('Housing.csv')

# Seleccionar las columnas numéricas para el análisis de correlación
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calcular la correlación de cada variable numérica con el precio
correlations = df[numerical_cols].corr()['price'].drop('price')
correlations_sorted = correlations.abs().sort_values(ascending=False)

# Variable con mayor correlación absoluta con price
best_feature = correlations_sorted.index[0]
print(f"Variable seleccionada para regresión simple: {best_feature}")

# Datos para regresión lineal simple
X_simple = df[[best_feature]].values
y = df['price'].values

# Entrenar el modelo de regresión lineal simple
model_simple = LinearRegression()
model_simple.fit(X_simple, y)

y_pred_simple = model_simple.predict(X_simple)
mse_simple = mean_squared_error(y, y_pred_simple)
r2_simple = r2_score(y, y_pred_simple)

print(f"MSE regresión simple: {mse_simple:.2f}")
print(f"R² regresión simple: {r2_simple:.4f}")

# Graficar datos reales y línea de tendencia de regresión simple
plt.figure(figsize=(8, 6))
plt.scatter(X_simple, y, color='blue', alpha=0.6, label='Datos reales')
plt.plot(X_simple, y_pred_simple, color='red', linewidth=2, label='Línea de tendencia')
plt.xlabel(best_feature)
plt.ylabel('price')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.savefig('grafico_regresion_simple.png', dpi=300)
plt.close()

# Seleccionar al menos 3 variables independientes para regresión lineal múltiple
# Se eligen las 3 variables numéricas con mayor correlación absoluta con price
best_multi_features = correlations_sorted.head(3).index.tolist()
X_multi = df[best_multi_features].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

y_pred_multi = model_multi.predict(X_multi)
mse_multi = mean_squared_error(y, y_pred_multi)
r2_multi = r2_score(y, y_pred_multi)

print('\nRegresión Lineal Múltiple:')
for feature_name, coef in zip(best_multi_features, model_multi.coef_):
    print(f"Coeficiente {feature_name}: {coef:.4f}")
print(f"MSE regresión múltiple: {mse_multi:.2f}")
print(f"R² regresión múltiple: {r2_multi:.4f}")

if r2_multi > r2_simple:
    mejor_modelo = 'Regresión Lineal Múltiple'
else:
    mejor_modelo = 'Regresión Lineal Simple'
print(f"Mejor modelo según R²: {mejor_modelo}")

# Regresión polinómica de grado 2 con la misma variable del modelo simple
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_transformer.fit_transform(X_simple)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

y_pred_poly = model_poly.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

print('\nRegresión Polinómica (grado 2):')
print(f"MSE regresión polinómica: {mse_poly:.2f}")
print(f"R² regresión polinómica: {r2_poly:.4f}")

# Ordenar los valores para graficar una curva suave
sorted_idx = X_simple[:, 0].argsort()
X_sorted = X_simple[sorted_idx]
y_poly_sorted = y_pred_poly[sorted_idx]

plt.figure(figsize=(8, 6))
plt.scatter(X_simple, y, color='blue', alpha=0.6, label='Datos reales')
plt.plot(X_sorted, y_poly_sorted, color='green', linewidth=2, label='Curva polinómica grado 2')
plt.xlabel(best_feature)
plt.ylabel('price')
plt.title('Regresión Polinómica Grado 2')
plt.legend()
plt.savefig('grafico_regresion_polinomica.png', dpi=300)
plt.close()

# Imprimir tabla resumen con los tres modelos
resumen = pd.DataFrame({
    'Modelo': ['Regresión Simple', 'Regresión Múltiple', 'Regresión Polinómica'],
    'MSE': [mse_simple, mse_multi, mse_poly],
    'R²': [r2_simple, r2_multi, r2_poly]
})

print('\nResumen de modelos:')
print(resumen.to_string(index=False, float_format='%.4f'))

# Comentario final sobre el ajuste del modelo polinómico
# El modelo polinómico se compara con el modelo lineal simple usando R². 
# Si R² del modelo polinómico es mayor, entonces el ajuste es mejor porque explica más varianza.
# Si R² es menor, entonces el modelo polinómico no mejora el ajuste respecto al lineal simple.
