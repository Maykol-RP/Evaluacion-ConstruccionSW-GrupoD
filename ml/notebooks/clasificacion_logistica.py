import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# =========================================
# 1. CARGAR DATASET
# =========================================
df = pd.read_csv("ml/data/healthcare_dataset.csv")

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
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# =========================================
# 9. PREDICCIÓN
# =========================================
y_pred = modelo.predict(X_test)

# =========================================
# 10. RESULTADOS
# =========================================
print("\n===== MATRIZ DE CONFUSIÓN =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== MÉTRICAS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))