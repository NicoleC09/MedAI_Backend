# 🏥 Sistema de Triage Asistido por IA
## Para Zonas de Alto Riesgo Sanitario en Colombia

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción

Sistema de Machine Learning para **clasificación automática de triage clínico**, entrenado con **100,000 casos reales** de pacientes en Chocó, Colombia. Diseñado para zonas con **severas limitaciones de infraestructura sanitaria**.

### 🎯 Objetivo Principal
**Minimizar el riesgo de clasificar erróneamente a un paciente en estado crítico (Falso Negativo)**, lo que en estas zonas implica un retraso mortal en la referencia hacia centros de mayor complejidad.

### 📊 Dataset
- **Fuente**: `triage_choco_100k_balanceado.csv` (100,000 casos reales)
- **Departamento**: Chocó, Colombia
- **Balanceado**: 20,000 casos por cada nivel de urgencia (1-5)
- **Características**: 15 variables clínicas por paciente

---

## 🏗️ Arquitectura del Sistema

```
ProyectoFinal/
├── config.py                      # Configuración global
├── main.py                        # Pipeline de entrenamiento (con dataset real)
├── requirements.txt               # Dependencias
├── README.md                      # Este archivo
├── triage_choco_100k_balanceado.csv  # Dataset real (100k casos)
│
├── src/                           # Módulos del sistema
│   ├── preprocessing.py           # Preprocesamiento + SMOTE
│   ├── model_training.py          # Entrenamiento XGBoost
│   ├── evaluation.py              # Evaluación y métricas
│   ├── interpretability.py        # Análisis SHAP
│   └── inference.py               # Motor de inferencia
│
├── models/
│   └── triage_model.joblib        # Modelo entrenado (3.8 MB)
│
├── outputs/                       # Reportes y gráficos
├── logs/                          # Logs de ejecución
└── venv/                          # Entorno virtual
```

---

## 📊 Clasificación de Triage

El modelo clasifica pacientes en **5 niveles** según el Sistema de Triage Colombiano/ESI:

| Nivel | Categoría | Descripción | Tiempo de Atención |
|-------|-----------|-------------|-------------------|
| 🔴 **1** | Resucitación | Riesgo vital inminente | Inmediato |
| 🟠 **2** | Emergencia | Condición grave | < 10 minutos |
| 🟡 **3** | Urgente | Urgencia moderada | < 30 minutos |
| 🟢 **4** | Menos Urgente | Urgencia menor | < 60 minutos |
| 🔵 **5** | No Urgente | Sin urgencia | Según disponibilidad |

---

## 🔧 Características Técnicas

### Features de Entrada
- **Signos Vitales**: FC, FR, PA, Temperatura, SpO2
- **Neurológico**: Escala de Glasgow (GCS)
- **Demográfico**: Edad, Sexo
- **Síntoma Principal**: Codificado (dolor torácico, disnea, trauma, etc.)
- **Comorbilidades**: Diabetes, HTA, cardiopatía, etc.

### Modelo
- **Algoritmo**: Gradient Boosting (XGBoost/LightGBM)
- **Manejo de Desbalance**: SMOTE + Pesos de clase personalizados
- **Interpretabilidad**: Valores SHAP para cada predicción

### Métricas de Éxito
- **Recall Nivel 1-2 ≥ 95%** (Objetivo crítico)
- F1-Score ponderado
- Matriz de confusión

---

## 🚀 Instalación

### 1. Clonar/Crear el proyecto
```bash
cd /path/to/ProyectoFinal
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Uso

### Ejecutar Pipeline de Entrenamiento

```bash
# Entrenar modelo con dataset real de 100k casos
python main.py
```

Esto:
1. ✅ Carga `triage_choco_100k_balanceado.csv`
2. ✅ Divide en train/validation/test
3. ✅ Preprocesa datos (normalización, imputación)
4. ✅ Entrena modelo XGBoost
5. ✅ Evalúa en conjunto test
6. ✅ Guarda modelo en `models/triage_model.joblib`

### Usar Modelo Preentrenado

```python
import joblib
import pandas as pd
from src.preprocessing import TriagePreprocessor

# Cargar modelo
model_package = joblib.load('models/triage_model.joblib')
model = model_package['model']
preprocessor = model_package['preprocessor']

# Preparar datos de nuevo paciente
X_nuevo = pd.DataFrame({
    'edad': [45],
    'frecuencia_cardiaca': [110],
    'frecuencia_respiratoria': [22],
    'temperatura': [38.5],
    'spO2': [92],
    'presion_sistolica': [130],
    'presion_diastolica': [85],
    'dolor': [7],
    # ... más features
})

# Preprocesar
X_procesado = preprocessor.transform(X_nuevo)

# Predecir
prediccion = model.predict(X_procesado)
probabilidades = model.predict_proba(X_procesado)

print(f"Nivel de Triage: {prediccion[0] + 1}")  # +1 porque es 0-indexed
print(f"Confianza: {max(probabilidades[0])*100:.1f}%")
```
print(f"Confianza: {result.confianza*100:.1f}%")
print(f"Alertas: {result.alertas}")
print(f"Recomendaciones: {result.recomendaciones}")
```

### API Rápida
```python
from src.inference import quick_triage

result = quick_triage(
    frecuencia_cardiaca=95,
    frecuencia_respiratoria=22,
    presion_sistolica=130,
    presion_diastolica=85,
    temperatura=37.8,
    saturacion_oxigeno=96,
    escala_glasgow=15,
    edad=45,
    sexo=1,
    sintoma_principal=5
)

print(result)
```

---

## 📈 Resultados Esperados

### Métricas de Evaluación

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| **Recall Nivel 1** | ≥ 95% | Detectar emergencias de resucitación |
| **Recall Nivel 2** | ≥ 95% | Detectar emergencias graves |
| **F1-Score Global** | ≥ 80% | Balance general del modelo |
| **Accuracy** | Informativo | No es métrica principal |

### Visualizaciones Generadas
- `confusion_matrix.png` - Matriz de confusión
- `class_metrics.png` - Métricas por clase
- `roc_curves.png` - Curvas ROC
- `shap_feature_importance.png` - Importancia SHAP
- `shap_summary.png` - Resumen de contribuciones

---

## 🔍 Interpretabilidad (SHAP)

El sistema incluye análisis SHAP para explicar cada predicción:

```python
from src.interpretability import TriageExplainer

explainer = TriageExplainer(model, feature_names)
explainer.fit(X_train)

# Explicación individual
explanation = explainer.explain_prediction(patient_data)
print(explanation['risk_factors'])
print(explanation['protective_factors'])

# Reporte clínico
report = explainer.generate_clinical_report(patient_data, patient_id="12345")
print(report)
```

---

## 📁 Archivos Generados

Después de ejecutar el pipeline:

```
models/
├── triage_model_final.joblib    # Modelo entrenado
├── preprocessor.joblib          # Preprocesador ajustado
└── triage_model_final.json      # Historial de entrenamiento

outputs/
├── confusion_matrix.png
├── class_metrics.png
├── roc_curves.png
├── pr_curves.png
├── shap_feature_importance.png
├── shap_summary.png
├── shap_critical_analysis.png
├── evaluation_metrics.json
└── pipeline_results.json

data/
├── triage_train.csv
└── triage_test.csv
```

---

## ⚠️ Consideraciones Éticas

> **IMPORTANTE**: Este sistema es una **herramienta de apoyo** a la decisión clínica, NO un reemplazo del criterio médico.

- La decisión final de triage debe ser tomada por personal médico calificado
- El modelo está optimizado para **minimizar falsos negativos** en emergencias
- Se recomienda validación clínica antes de despliegue en producción
- Los datos sintéticos deben ser reemplazados por datos reales validados

---

## 🔄 Próximos Pasos

1. **Validación Clínica**: Evaluar con datos reales de urgencias
2. **Integración**: Desarrollar API REST para tele-triage
3. **Monitoreo**: Implementar drift detection en producción
4. **Expansión**: Incluir más features (laboratorios, imágenes)

---

## 📚 Referencias

- Sistema de Triage Colombiano (Ministerio de Salud)
- Emergency Severity Index (ESI) - AHRQ
- XGBoost: A Scalable Tree Boosting System
- SHAP: A Unified Approach to Interpreting Model Predictions

---

## 👥 Autores

Proyecto desarrollado para el curso de PTIA - Proyecto Final

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
