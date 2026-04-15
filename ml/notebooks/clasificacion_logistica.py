import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 1. CARGAR DATASET
# =========================================
script_dir = Path(__file__).resolve().parent
csv_path = script_dir.parent / "data" / "healthcare_dataset.csv"
if not csv_path.exists():
    csv_path = script_dir.parent.parent / "healthcare_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {script_dir.parent / 'data' / 'healthcare_dataset.csv'} ni en {csv_path}")

df = pd.read_csv(csv_path)

print("Columnas del dataset:")
print(df.columns.tolist())

# =========================================
# 2. LIMPIEZA BÁSICA
# =========================================
df = df.dropna()

# =========================================
# 3. DEFINIR VARIABLE OBJETIVO
#    Usaremos "Test Results" como variable categórica
#    La convertimos a binaria:
#    Abnormal = 1
#    Normal / Inconclusive = 0
# =========================================
if "Test Results" not in df.columns:
    raise ValueError("No se encontró la columna 'Test Results' en el dataset.")

df["Test Results"] = df["Test Results"].astype(str).str.strip()

df["Target"] = df["Test Results"].apply(lambda x: 1 if x.lower() == "abnormal" else 0)

# =========================================
# 4. SELECCIÓN DE VARIABLES INDEPENDIENTES
#    Se usan columnas que suelen existir en este dataset
# =========================================
feature_columns = [
    "Age",
    "Gender",
    "Billing Amount",
    "Room Number",
    "Admission Type"
]

# Verificar que existan
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Faltan estas columnas en el dataset: {missing_cols}")

X = df[feature_columns].copy()
y = df["Target"]

# =========================================
# 5. PREPROCESAMIENTO
#    Convertir variables categóricas a numéricas
# =========================================
X = pd.get_dummies(X, drop_first=True)

# =========================================
# 6. ESCALADO
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================
# 7. DIVISIÓN DE DATOS
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================================
# 8. MODELO DE REGRESIÓN LOGÍSTICA
# =========================================
modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
modelo.fit(X_train, y_train)

# =========================================
# 10. COEFICIENTES
# =========================================
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_[0]
})
print("\n===== COEFICIENTES DEL MODELO =====")
print(coef_df.to_string(index=False, float_format='%.6f'))

# =========================================
# 11. PREDICCIÓN
# =========================================
y_pred = modelo.predict(X_test)

# =========================================
# 12. RESULTADOS
# =========================================
print("\n===== MATRIZ DE CONFUSIÓN =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n===== MÉTRICAS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))

# =========================================
# 13. VISUALIZACIÓN
# =========================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal/Inconclusive', 'Abnormal'], 
            yticklabels=['Normal/Inconclusive', 'Abnormal'])
plt.title('Matriz de Confusión - Clasificación Logística')
plt.ylabel('Valor Verdadero')
plt.xlabel('Valor Predicho')
output_path = script_dir / 'confusion_matrix.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
print(f"\nGráfico guardado como '{output_path}'")

# =========================================
# CONCLUSIÓN
# =========================================
# El valor de Accuracy indica qué proporción de predicciones fue correcta sobre el conjunto de prueba.
# Un valor alto sugiere que el modelo clasifica bien en general, pero debe evaluarse también junto a precisión y recall.
# En este caso, si el accuracy es elevado y las métricas de precisión y recall son razonables, el modelo puede considerarse bueno,
# aunque siempre hay que revisar si existen falsos positivos o falsos negativos importantes.
# Se usó class_weight="balanced" porque las clases pueden estar desbalanceadas en el dataset.
# Este parámetro ajusta automáticamente el peso de cada clase en la función de pérdida,
# para evitar que el modelo favorezca únicamente a la clase mayoritaria.
