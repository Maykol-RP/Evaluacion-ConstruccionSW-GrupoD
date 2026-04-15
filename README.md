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

Este repositorio contiene una evaluación práctica de Machine Learning con un enfoque colaborativo usando Git. El proyecto incluye un script de regresión sobre el dataset `Housing.csv`, una implementación de regresión logística en `ml/notebooks` y demuestra el flujo de trabajo de trabajo en equipo con ramas para diferentes funcionalidades.

---

## 🚀 Cómo ejecutar el código

1. Clonar el repositorio.
2. Instalar las dependencias:

```bash
pip install pandas scikit-learn matplotlib numpy seaborn
```

3. Ejecutar los scripts de los modelos:

```bash
python regresion_modelos.py
python ml/notebooks/clasificacion_logistica.py
```

4. Revisar los archivos generados:

- `grafico_regresion_simple.png`
- `grafico_regresion_polinomica.png`
- `ml/notebooks/confusion_matrix.png`

---

## 📊 Tabla de métricas

| Modelo | R² | MSE | Accuracy |
|---|---|---|---|
| Regresión Lineal Simple (var: area) | 0.2873 | 2,488,861,398,180.66 | — |
| Regresión Lineal Múltiple | 0.5309 | 1,638,064,785,021.31 | — |
| Regresión Polinómica Grado 2 | 0.3231 | 2,363,900,046,361.73 | — |
| Regresión Logística | N/A | N/A | 0.5005 |

> Nota: La regresión logística utiliza métricas de clasificación, por lo que el `Accuracy` es la medida principal aquí.

---

## 🌿 Estrategia de ramas

- `main`: rama protegida para el código estable y versiones de producción.
- `develop`: rama de integración donde se consolidan las nuevas funcionalidades.
- `feat/regresion-modelos`: rama de trabajo para la Parte 2, enfocada en regresiones lineales y polinómicas.
- `feat/regresion-logistica`: rama de trabajo para la Parte 3, orientada a la implementación de regresión logística y clasificación.

---

## 📝 Conclusión

El mejor rendimiento entre los modelos de regresión fue de la Regresión Lineal Múltiple con R²=0.5309, ya que al usar varias variables independientes captura mejor la variabilidad del precio que un modelo simple basado solamente en área. Esto indica que la combinación de características aporta más información predictiva en este dataset.

La Regresión Polinómica Grado 2 no mejoró lo suficiente respecto al modelo múltiple, lo que sugiere que la relación entre precio y área no es lo bastante no lineal para justificar mayor complejidad. La Regresión Logística, por su parte, es un modelo de clasificación diferente y su Accuracy de 0.5005 muestra un desempeño limitado para los resultados médicos del conjunto de datos.
