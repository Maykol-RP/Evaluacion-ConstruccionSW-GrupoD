# 🏠 Evaluacion-ConstruccionSW-GrupoD

## 👥 Integrantes

| # | Estudiante | Carrera |
|---|-----------|---------|
| 1 | **Fabrizio Aguilar** | 🎓 Ingeniería de Sistemas |
| 2 | **Mariano Laura** | 🎓 Ingeniería de Sistemas |
| 3 | **Maykol Rocca** | 🎓 Ingeniería de Sistemas |
| 4 | **Fray Salcedo** | 🎓 Ingeniería de Sistemas |
| 5 | **Liz Vargas** | 🎓 Ingeniería de Sistemas |

---

## 📋 Descripción

Este repositorio contiene una evaluación práctica de Machine Learning con un enfoque colaborativo usando Git. El proyecto incluye un script de regresión sobre el dataset `Housing.csv` y demuestra el flujo de trabajo de trabajo en equipo con ramas para diferentes funcionalidades.

---

## 🚀 Cómo ejecutar el código

1. Clonar el repositorio.
2. Instalar las dependencias:

```bash
pip install pandas scikit-learn matplotlib numpy
```

3. Ejecutar el script:

```bash
python regresion_modelos.py
```

---

## 📊 Tabla de métricas

| Modelo | R² | MSE |
|---|---|---|
| Regresión Lineal Simple (var: area) | 0.2873 | 2,488,861,398,180.66 |
| Regresión Lineal Múltiple | 0.5309 | 1,638,064,785,021.31 |
| Regresión Polinómica Grado 2 | 0.3231 | 2,363,900,046,361.73 |
| Regresión Logística | [PENDIENTE] | — |

---

## 🌿 Estrategia de ramas

- `main`: rama protegida para el código estable y versiones de producción.
- `develop`: rama de integración donde se consolidan las nuevas funcionalidades.
- `feat/regresion-modelos`: rama de trabajo para la Parte 2, enfocada en regresiones lineales y polinómicas.
- `feat/regresion-logistica`: rama de trabajo para la Parte 3, orientada a la implementación de regresión logística.

---

## 📝 Conclusión

El mejor modelo fue la Regresión Lineal Múltiple con R²=0.5309 porque al usar múltiples variables captura más información que la simple. Esto le permite explicar mejor la variación del precio al considerar distintos factores simultáneamente.

La regresión polinómica no superó a la múltiple porque la relación entre precio y área no es suficientemente no-lineal para justificar la complejidad adicional. En este dataset, el modelo múltiple ofrece un mejor balance entre simplicidad y capacidad predictiva.
