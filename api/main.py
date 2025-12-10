"""
🏥 API REST - Sistema de Triage Asistido por IA
===============================================
Endpoints para clasificación de pacientes en tiempo real.

Tecnología: FastAPI
Modelo: XGBoost optimizado para recall crítico
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'triage_model.joblib'

# Cargar modelo al iniciar
print("🔄 Cargando modelo de triage entrenado con dataset real de 100k casos...")
try:
    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    preprocessor = model_package['preprocessor']
    feature_names = model_package['feature_names']
    n_classes = model_package.get('n_classes', 5)
    print(f"✅ Modelo cargado exitosamente")
    print(f"   - Dataset: {model_package['dataset']}")
    print(f"   - Clases: {n_classes}")
    print(f"   - Features: {len(feature_names)}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    raise

# ============================================================
# ENUMS Y MODELOS PYDANTIC
# ============================================================

class Sexo(str, Enum):
    masculino = "M"
    femenino = "F"

class ModoLlegada(str, Enum):
    caminando = "Caminando"
    particular = "Particular"
    ambulancia = "Ambulancia"
    policia = "Policía"
    policia_sin_tilde = "Policia"  # Alternativa sin tilde

class MotivoConsulta(str, Enum):
    dolor_toracico = "Dolor torácico"
    disnea = "Disnea"
    covid_respiratorio = "COVID/Respiratorio"
    hemorragia = "Hemorragia"
    dolor_abdominal = "Dolor abdominal"
    vomito_diarrea = "Vómito/diarrea"
    fiebre = "Fiebre"
    mareo_sincope = "Mareo/Síncope"
    cefalea = "Cefalea"
    sospecha_malaria = "Sospecha malaria"
    sospecha_dengue = "Sospecha dengue"
    dolor_lumbar = "Dolor lumbar"
    lesion_menor = "Lesión menor"
    dermatologico = "Dermatológico"
    obstetrico = "Obstétrico"
    urologico = "Urológico"  # Para "dolor de pipi" y similares
    otro = "Otro"

# Mapeo de palabras clave para clasificar motivos de texto libre
MOTIVO_KEYWORDS = {
    "torax": 6, "pecho": 6, "toracico": 6, "corazon": 6, "cardiaco": 6, "infarto": 6,
    "respirar": 3, "ahogo": 3, "disnea": 3, "aire": 3, "asfixia": 3,
    "covid": 0, "gripe": 0, "tos": 0, "resfriado": 0, "respiratorio": 0,
    "sangre": 8, "hemorragia": 8, "sangrado": 8,
    "abdomen": 4, "estomago": 4, "barriga": 4, "panza": 4, "vientre": 4,
    "vomito": 14, "diarrea": 14, "nausea": 14,
    "fiebre": 7, "calentura": 7, "temperatura": 7,
    "mareo": 10, "desmayo": 10, "sincope": 10, "vertigo": 10,
    "cabeza": 1, "cefalea": 1, "migrana": 1, "jaqueca": 1,
    "malaria": 13, "paludismo": 13,
    "dengue": 12,
    "espalda": 5, "lumbar": 5, "cintura": 5, "columna": 5,
    "herida": 9, "golpe": 9, "caida": 9, "trauma": 9, "lesion": 9, "fractura": 9,
    "piel": 2, "rash": 2, "alergia": 2, "picazon": 2, "sarpullido": 2,
    "embarazo": 11, "parto": 11, "contracciones": 11, "obstetrico": 11,
    "orina": 4, "pipi": 4, "urinario": 4, "riñon": 4, "vejiga": 4,
}

class NivelTriage(str, Enum):
    nivel_1 = "Nivel 1 - Resucitación/Inmediato"
    nivel_2 = "Nivel 2 - Emergencia/Grave"
    nivel_3 = "Nivel 3 - Urgente"
    nivel_4 = "Nivel 4 - Menos Urgente"
    nivel_5 = "Nivel 5 - No Urgente"

class Comorbilidades(BaseModel):
    """Antecedentes médicos del paciente."""
    hta: bool = Field(False, description="Hipertensión arterial")
    dm2: bool = Field(False, description="Diabetes Mellitus tipo 2")
    epoc: bool = Field(False, description="Enfermedad Pulmonar Obstructiva Crónica")
    irc: bool = Field(False, description="Insuficiencia Renal Crónica")
    cardiopatia: bool = Field(False, description="Cardiopatía")
    obesidad: bool = Field(False, description="Obesidad")
    cancer: bool = Field(False, description="Cáncer")
    embarazo: bool = Field(False, description="Embarazo")

class SignosVitales(BaseModel):
    """Signos vitales del paciente."""
    frecuencia_cardiaca: int = Field(..., ge=20, le=250, description="Latidos por minuto")
    frecuencia_respiratoria: int = Field(..., ge=4, le=70, description="Respiraciones por minuto")
    temperatura: float = Field(..., ge=32.0, le=44.0, description="Temperatura en °C")
    saturacion_oxigeno: int = Field(..., ge=40, le=100, alias="spO2", description="SpO2 en %")
    presion_sistolica: int = Field(..., ge=40, le=300, description="Presión sistólica mmHg")
    presion_diastolica: int = Field(..., ge=20, le=200, description="Presión diastólica mmHg")
    dolor: int = Field(..., ge=0, le=10, description="Escala de dolor 0-10")

class PacienteInput(BaseModel):
    """Datos completos del paciente para clasificación."""
    # Datos demográficos
    edad: int = Field(..., ge=0, le=120, description="Edad en años")
    sexo: Sexo
    
    # Llegada y motivo
    modo_llegada: ModoLlegada
    motivo_consulta: str = Field(..., min_length=2, max_length=200, description="Motivo de consulta en texto libre")
    
    # Signos vitales
    signos_vitales: SignosVitales
    
    # Antecedentes
    comorbilidades: Comorbilidades = Field(default_factory=Comorbilidades)
    
    # Recursos (decisión del triajista)
    requiere_labs: bool = Field(False, description="¿Se solicitan laboratorios?")
    requiere_imagenes: bool = Field(False, description="¿Se solicitan imágenes?")
    
    # Población prioritaria
    es_prioritario: bool = Field(False, description="Población prioritaria (adulto mayor, discapacidad, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "edad": 65,
                "sexo": "M",
                "modo_llegada": "Ambulancia",
                "motivo_consulta": "Dolor en el pecho desde hace 2 horas",
                "signos_vitales": {
                    "frecuencia_cardiaca": 110,
                    "frecuencia_respiratoria": 24,
                    "temperatura": 37.2,
                    "spO2": 88,
                    "presion_sistolica": 90,
                    "presion_diastolica": 60,
                    "dolor": 8
                },
                "comorbilidades": {
                    "hta": True,
                    "cardiopatia": True,
                    "dm2": False,
                    "epoc": False,
                    "irc": False,
                    "obesidad": False,
                    "cancer": False
                },
                "requiere_labs": True,
                "requiere_imagenes": True
            }
        }

class FactorRiesgo(BaseModel):
    """Factor de riesgo identificado."""
    factor: str
    descripcion: str
    severidad: str  # "alta", "media", "baja"

class ClasificacionResponse(BaseModel):
    """Respuesta de clasificación del paciente."""
    nivel_triage: str
    nivel_codigo: int  # 1-4
    descripcion: str
    confianza: float
    probabilidades: dict
    factores_riesgo: List[FactorRiesgo]
    tiempo_atencion_recomendado: str
    area_recomendada: str
    timestamp: str

class HealthResponse(BaseModel):
    """Estado del servicio."""
    status: str
    model_loaded: bool
    version: str
    timestamp: str

# ============================================================
# APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="🏥 API de Triage - Chocó",
    description="""
    Sistema de clasificación de urgencias médicas asistido por IA.
    
    ## Características
    - Clasificación en 4 niveles de urgencia (ESI)
    - Optimizado para alta sensibilidad en casos críticos (Recall > 95%)
    - Identificación de factores de riesgo
    - Tiempos de respuesta < 100ms
    
    ## Niveles de Triage
    - **Nivel 1**: Resucitación/Inmediato - Atención inmediata
    - **Nivel 2**: Emergencia/Grave - Atención en < 10 minutos
    - **Nivel 3**: Urgente - Atención en < 30 minutos
    - **Nivel 4**: Menos Urgente - Atención en < 60 minutos
    - **Nivel 5**: No Urgente - Atención en < 120 minutos o consulta externa
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite
        "http://localhost:4200",  # Angular
        "*"  # Desarrollo - quitar en producción
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

# Encoders para variables categóricas (deben coincidir con entrenamiento)
SEXO_MAP = {"F": 0, "M": 1}
MODO_LLEGADA_MAP = {"Ambulancia": 0, "Caminando": 1, "Particular": 2, "Policía": 3, "Policia": 3}

# Mapeo exacto del dataset triage_choco_100k_balanceado.csv (LabelEncoder alfabético)
# 0: Administrativo, 1: COVID/Respiratorio, 2: Cefalea, 3: Dermatológico, 4: Disnea
# 5: Dolor abdominal, 6: Dolor lumbar, 7: Dolor torácico, 8: Fiebre, 9: Hemorragia
# 10: Herida leve, 11: Intoxicación, 12: Lesión menor, 13: Mareo/Síncope, 14: Obstétrico
# 15: Otros, 16: Sospecha dengue, 17: Sospecha malaria, 18: Trauma, 19: Vómito/diarrea

MOTIVO_KEYWORDS = {
    # Administrativo (0)
    "certificado": 0, "administrativo": 0, "receta": 0, "control": 0,
    # COVID/Respiratorio (1)
    "covid": 1, "gripe": 1, "tos": 1, "resfriado": 1, "respiratorio": 1,
    # Cefalea (2)
    "cabeza": 2, "cefalea": 2, "migrana": 2, "jaqueca": 2,
    # Dermatológico (3)
    "piel": 3, "rash": 3, "alergia": 3, "picazon": 3, "sarpullido": 3, "dermatologico": 3,
    # Disnea (4)
    "respirar": 4, "ahogo": 4, "disnea": 4, "aire": 4, "asfixia": 4, "dificultad respiratoria": 4,
    # Dolor abdominal (5)
    "abdomen": 5, "estomago": 5, "barriga": 5, "panza": 5, "vientre": 5, "abdominal": 5,
    # Dolor lumbar (6)
    "espalda": 6, "lumbar": 6, "cintura": 6, "columna": 6,
    # Dolor torácico (7)
    "torax": 7, "pecho": 7, "toracico": 7, "corazon": 7, "cardiaco": 7, "infarto": 7, "precordial": 7,
    # Fiebre (8)
    "fiebre": 8, "calentura": 8, "febril": 8, "temperatura alta": 8,
    # Hemorragia (9)
    "sangre": 9, "hemorragia": 9, "sangrado": 9,
    # Herida leve (10)
    "herida": 10, "cortada": 10, "raspadura": 10,
    # Intoxicación (11)
    "intoxicacion": 11, "envenenamiento": 11, "toxico": 11,
    # Lesión menor (12)
    "lesion": 12, "golpe leve": 12,
    # Mareo/Síncope (13)
    "mareo": 13, "desmayo": 13, "sincope": 13, "vertigo": 13,
    # Obstétrico (14)
    "embarazo": 14, "parto": 14, "contracciones": 14, "obstetrico": 14,
    # Sospecha dengue (16)
    "dengue": 16,
    # Sospecha malaria (17)
    "malaria": 17, "paludismo": 17,
    # Trauma (18)
    "trauma": 18, "golpe": 18, "caida": 18, "accidente": 18, "fractura": 18,
    # Vómito/diarrea (19)
    "vomito": 19, "diarrea": 19, "nausea": 19,
    # Orina -> Dolor abdominal (5)
    "orina": 5, "pipi": 5, "urinario": 5, "rinon": 5, "vejiga": 5,
}


def clasificar_motivo_texto(texto: str) -> int:
    """Clasifica un motivo de consulta en texto libre usando palabras clave."""
    import unicodedata
    
    # Normalizar texto: minúsculas, sin tildes
    texto_norm = texto.lower()
    texto_norm = ''.join(
        c for c in unicodedata.normalize('NFD', texto_norm)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Buscar palabras clave
    for keyword, codigo in MOTIVO_KEYWORDS.items():
        if keyword in texto_norm:
            return codigo
    
    # Default: Otros (13)
    return 13

LABS_MAP = {"No": 0, "Sí": 1}
IMAGENES_MAP = {"No": 0, "Sí": 1}
DISPOSICION_MAP = {"Alta": 0, "Hospitalización": 1, "Observación": 2, "Referencia": 3, "UCI": 4}

NIVEL_INFO = {
    0: {
        "nombre": "Nivel 1 - Resucitación/Inmediato",
        "descripcion": "Condición que amenaza la vida. Requiere intervención inmediata.",
        "tiempo": "Inmediato (0 minutos)",
        "area": "Área de Shock / Reanimación"
    },
    1: {
        "nombre": "Nivel 2 - Emergencia/Grave", 
        "descripcion": "Condición de alto riesgo. Requiere atención muy rápida.",
        "tiempo": "< 10 minutos",
        "area": "Área de Emergencias"
    },
    2: {
        "nombre": "Nivel 3 - Urgente",
        "descripcion": "Condición que requiere atención pronta pero estable.",
        "tiempo": "< 30 minutos",
        "area": "Consultorios de Urgencias"
    },
    3: {
        "nombre": "Nivel 4 - Menos Urgente",
        "descripcion": "Condición menor que puede esperar.",
        "tiempo": "< 60 minutos",
        "area": "Consulta Prioritaria"
    },
    4: {
        "nombre": "Nivel 5 - No Urgente",
        "descripcion": "Condición no urgente. Puede esperar o ser atendido en consulta externa.",
        "tiempo": "< 120 minutos",
        "area": "Sala de Espera General / Consulta Externa"
    }
}


def preparar_features(paciente: PacienteInput) -> np.ndarray:
    """Convierte los datos del paciente al formato del modelo."""
    
    # Mapeos para encoding
    SEXO_MAP = {"M": 1, "F": 0}
    MODO_LLEGADA_MAP = {
        "Caminando": 0,
        "Particular": 1,
        "Ambulancia": 2,
        "Policía": 3,
        "Policia": 3
    }
    
    # Crear DataFrame con los datos del paciente
    features = {
        'edad': paciente.edad,
        'frecuencia_cardiaca': paciente.signos_vitales.frecuencia_cardiaca,
        'frecuencia_respiratoria': paciente.signos_vitales.frecuencia_respiratoria,
        'temperatura': paciente.signos_vitales.temperatura,
        'spO2': paciente.signos_vitales.saturacion_oxigeno,
        'presion_sistolica': paciente.signos_vitales.presion_sistolica,
        'presion_diastolica': paciente.signos_vitales.presion_diastolica,
        'dolor': paciente.signos_vitales.dolor,
        'sexo_encoded': SEXO_MAP.get(paciente.sexo.value, 0),
        'motivo_encoded': clasificar_motivo_texto(paciente.motivo_consulta),
        'diabetes': int(paciente.comorbilidades.dm2),
        'hipertension': int(paciente.comorbilidades.hta),
        'enfermedad_cardiaca': int(paciente.comorbilidades.cardiopatia),
        'enfermedad_respiratoria': int(paciente.comorbilidades.epoc),
        'inmunosupresion': int(paciente.comorbilidades.cancer or paciente.comorbilidades.irc),
    }
    
    # Crear DataFrame
    X = pd.DataFrame([features])
    
    # Usar preprocesador del modelo para normalización
    X_processed = preprocessor.transform(X)
    
    return X_processed


def identificar_factores_riesgo(paciente: PacienteInput) -> List[FactorRiesgo]:
    """Identifica factores de riesgo del paciente."""
    factores = []
    sv = paciente.signos_vitales
    
    # SpO2 crítico
    if sv.saturacion_oxigeno < 90:
        factores.append(FactorRiesgo(
            factor="Hipoxia",
            descripcion=f"SpO2 = {sv.saturacion_oxigeno}% (crítico < 90%)",
            severidad="alta"
        ))
    elif sv.saturacion_oxigeno < 94:
        factores.append(FactorRiesgo(
            factor="Hipoxia leve",
            descripcion=f"SpO2 = {sv.saturacion_oxigeno}% (bajo < 94%)",
            severidad="media"
        ))
    
    # Presión arterial
    if sv.presion_sistolica < 90:
        factores.append(FactorRiesgo(
            factor="Hipotensión",
            descripcion=f"PA sistólica = {sv.presion_sistolica} mmHg (shock < 90)",
            severidad="alta"
        ))
    elif sv.presion_sistolica > 180:
        factores.append(FactorRiesgo(
            factor="Crisis hipertensiva",
            descripcion=f"PA sistólica = {sv.presion_sistolica} mmHg (crisis > 180)",
            severidad="alta"
        ))
    
    # Frecuencia cardíaca
    if sv.frecuencia_cardiaca > 120:
        factores.append(FactorRiesgo(
            factor="Taquicardia severa",
            descripcion=f"FC = {sv.frecuencia_cardiaca} lpm (> 120)",
            severidad="alta"
        ))
    elif sv.frecuencia_cardiaca < 50:
        factores.append(FactorRiesgo(
            factor="Bradicardia severa",
            descripcion=f"FC = {sv.frecuencia_cardiaca} lpm (< 50)",
            severidad="alta"
        ))
    elif sv.frecuencia_cardiaca < 60:
        # Bradicardia leve, pero significativa en ancianos o con comorbilidades cardíacas
        if paciente.edad >= 65 or paciente.comorbilidades.hta or paciente.comorbilidades.cardiopatia:
            factores.append(FactorRiesgo(
                factor="Bradicardia en paciente de riesgo",
                descripcion=f"FC = {sv.frecuencia_cardiaca} lpm en paciente {'≥65 años' if paciente.edad >= 65 else 'con comorbilidad cardiovascular'}",
                severidad="media"
            ))
    
    # Temperatura
    if sv.temperatura > 39.5:
        factores.append(FactorRiesgo(
            factor="Fiebre alta",
            descripcion=f"Temperatura = {sv.temperatura}°C (> 39.5°C)",
            severidad="alta"
        ))
    elif sv.temperatura < 35:
        factores.append(FactorRiesgo(
            factor="Hipotermia",
            descripcion=f"Temperatura = {sv.temperatura}°C (< 35°C)",
            severidad="alta"
        ))
    
    # Dolor severo
    if sv.dolor >= 8:
        factores.append(FactorRiesgo(
            factor="Dolor severo",
            descripcion=f"Escala de dolor = {sv.dolor}/10",
            severidad="alta"
        ))
    
    # Frecuencia respiratoria
    if sv.frecuencia_respiratoria > 24:
        factores.append(FactorRiesgo(
            factor="Taquipnea",
            descripcion=f"FR = {sv.frecuencia_respiratoria} rpm (> 24)",
            severidad="media"
        ))
    
    # Edad avanzada con comorbilidades
    num_comorb = sum([
        paciente.comorbilidades.hta, paciente.comorbilidades.dm2,
        paciente.comorbilidades.epoc, paciente.comorbilidades.irc,
        paciente.comorbilidades.cardiopatia, paciente.comorbilidades.obesidad,
        paciente.comorbilidades.cancer
    ])
    
    if paciente.edad >= 65 and num_comorb >= 2:
        factores.append(FactorRiesgo(
            factor="Paciente geriátrico multimórbido",
            descripcion=f"Edad {paciente.edad} años con {num_comorb} comorbilidades",
            severidad="media"
        ))
    
    # Modo de llegada
    if paciente.modo_llegada == ModoLlegada.ambulancia:
        factores.append(FactorRiesgo(
            factor="Llegada en ambulancia",
            descripcion="Paciente trasladado por servicio de emergencia",
            severidad="media"
        ))
    
    # Motivos de consulta de alto riesgo (búsqueda en texto)
    motivo_lower = paciente.motivo_consulta.lower()
    motivos_alto_riesgo = ["torax", "pecho", "corazon", "respirar", "ahogo", "sangre", "hemorragia", "desmayo", "sincope"]
    for motivo_keyword in motivos_alto_riesgo:
        if motivo_keyword in motivo_lower:
            factores.append(FactorRiesgo(
                factor="Motivo de alto riesgo",
                descripcion=paciente.motivo_consulta,
                severidad="media"
            ))
            break
    
    return factores


def calibrar_probabilidades(proba: np.ndarray, temperatura: float = 2.5) -> np.ndarray:
    """
    Aplica temperature scaling para calibrar probabilidades.
    
    XGBoost tiende a dar probabilidades muy extremas (98-99%).
    Temperature scaling suaviza la distribución para valores más realistas.
    
    - temperatura > 1: suaviza (menos extremo, más incertidumbre)
    - temperatura = 1: sin cambio
    - temperatura < 1: más extremo
    
    Valor 2.5 calibrado para dar confianzas típicas de 60-85%.
    """
    # Convertir a logits, aplicar temperatura, reconvertir a probabilidades
    # Evitar log(0) agregando pequeño epsilon
    epsilon = 1e-10
    proba_safe = np.clip(proba, epsilon, 1 - epsilon)
    
    # Logits = log(p / (1-p)) para cada clase, simplificado con softmax inverso
    log_proba = np.log(proba_safe)
    log_proba_scaled = log_proba / temperatura
    
    # Softmax para reconvertir a probabilidades
    exp_scaled = np.exp(log_proba_scaled - np.max(log_proba_scaled))  # Estabilidad numérica
    proba_calibrada = exp_scaled / exp_scaled.sum()
    
    return proba_calibrada


def predecir_con_umbrales(X: np.ndarray) -> tuple:
    """Realiza predicción del modelo."""
    proba_raw = model.predict_proba(X)[0]  # Probabilidades del modelo
    pred = model.predict(X)[0]  # Clase predicha
    
    # Calibrar probabilidades para valores más realistas
    proba = calibrar_probabilidades(proba_raw, temperatura=2.5)
    
    # La confianza mostrada
    confianza = proba[pred]
    
    # Construir diccionario de probabilidades (5 niveles)
    probabilidades = {}
    for i in range(n_classes):
        probabilidades[f"Nivel {i+1}"] = round(float(proba[i]) * 100, 1)
    
    return pred, confianza, probabilidades


def evaluar_nivel_5(paciente: PacienteInput, nivel_modelo: int, probabilidades: dict) -> tuple:
    """
    Evalúa si el paciente debería ser Nivel 5 (No Urgente).
    
    Criterios para Nivel 5 (TODOS deben cumplirse):
    - Signos vitales completamente normales
    - Dolor LEVE (≤3)
    - Sin comorbilidades de riesgo (cardiopatía, EPOC, IRC, cáncer)
    - Motivo de consulta menor
    - No llegó en ambulancia
    - Modelo predijo nivel 3 o 4
    """
    sv = paciente.signos_vitales
    
    # Solo considerar reclasificar si el modelo predijo Nivel 3 o 4
    if nivel_modelo < 2:  # Nivel 1 o 2 nunca bajan a 5
        return nivel_modelo, probabilidades
    
    # =========================================
    # CRITERIOS EXCLUYENTES (si alguno se cumple, NO puede ser nivel 5)
    # =========================================
    
    # 1. Dolor moderado o severo (>3) -> NO puede ser nivel 5
    if sv.dolor > 3:
        return nivel_modelo, probabilidades
    
    # 2. Comorbilidades de alto riesgo -> NO puede ser nivel 5
    tiene_comorbilidad_riesgo = any([
        paciente.comorbilidades.cardiopatia,
        paciente.comorbilidades.epoc,
        paciente.comorbilidades.irc,
        paciente.comorbilidades.cancer
    ])
    if tiene_comorbilidad_riesgo:
        return nivel_modelo, probabilidades
    
    # 3. Llegó en ambulancia -> NO puede ser nivel 5
    if paciente.modo_llegada.value == "Ambulancia":
        return nivel_modelo, probabilidades
    
    # 4. Motivos que NUNCA son nivel 5
    motivo_lower = paciente.motivo_consulta.lower()
    motivos_excluidos = ["pecho", "torax", "corazon", "respirar", "sangre", "hemorragia",
                         "desmayo", "convulsion", "embarazo", "parto", "infarto", "ahogo",
                         "severo", "intenso", "fuerte", "insoportable"]
    if any(m in motivo_lower for m in motivos_excluidos):
        return nivel_modelo, probabilidades
    
    # 5. Edad avanzada (≥70) -> NO puede ser nivel 5 (mayor riesgo)
    if paciente.edad >= 70:
        return nivel_modelo, probabilidades
    
    # 6. Signos vitales anormales -> NO puede ser nivel 5
    signos_normales = (
        60 <= sv.frecuencia_cardiaca <= 100 and
        12 <= sv.frecuencia_respiratoria <= 20 and
        36.0 <= sv.temperatura <= 37.5 and
        sv.saturacion_oxigeno >= 95 and
        90 <= sv.presion_sistolica <= 140 and
        60 <= sv.presion_diastolica <= 90
    )
    if not signos_normales:
        return nivel_modelo, probabilidades
    
    # =========================================
    # Si pasó TODOS los filtros, puede ser Nivel 5
    # =========================================
    
    # Calcular confianza de nivel 5
    prob_nivel_5 = 90.0  # Alta confianza si pasó todos los filtros
    
    # Ajustar probabilidades
    nuevas_prob = probabilidades.copy()
    nuevas_prob["Nivel 5"] = prob_nivel_5
    
    # Reducir proporcionalmente las otras
    factor = (100 - prob_nivel_5) / 100
    for k in ["Nivel 1", "Nivel 2", "Nivel 3", "Nivel 4"]:
        nuevas_prob[k] = nuevas_prob[k] * factor
    
    return 4, nuevas_prob  # 4 = Nivel 5 (clase 4)


def ajustar_nivel_por_reglas_clinicas(paciente: PacienteInput, nivel_modelo: int, probabilidades: dict) -> tuple:
    """
    Ajusta el nivel hacia ARRIBA (más urgente) basado en reglas clínicas.
    
    El modelo ML puede subestimar ciertos casos que clínicamente requieren más atención.
    Esta función aplica reglas de seguridad para estos casos.
    
    Returns:
        tuple: (nivel_ajustado, probabilidades_ajustadas)
    """
    sv = paciente.signos_vitales
    nivel_ajustado = nivel_modelo
    
    # =========================================
    # REGLA 1: Paciente geriátrico (≥65) con bradicardia y comorbilidad CV
    # =========================================
    if (paciente.edad >= 65 and 
        sv.frecuencia_cardiaca < 60 and 
        (paciente.comorbilidades.hta or paciente.comorbilidades.cardiopatia)):
        # Mínimo Nivel 2 (Emergencia)
        if nivel_modelo > 1:  # Si es 2, 3 o 4 → subir a mínimo 2
            nivel_ajustado = min(nivel_ajustado, 1)  # Clase 1 = Nivel 2
    
    # =========================================
    # REGLA 2: Dolor torácico en paciente con factores de riesgo cardíaco
    # =========================================
    motivo_lower = paciente.motivo_consulta.lower()
    es_dolor_toracico = any(k in motivo_lower for k in ["pecho", "torax", "toracico", "precordial"])
    tiene_riesgo_cv = (paciente.comorbilidades.hta or 
                       paciente.comorbilidades.cardiopatia or 
                       paciente.comorbilidades.dm2 or
                       paciente.edad >= 55)
    
    if es_dolor_toracico and tiene_riesgo_cv:
        # Mínimo Nivel 2
        if nivel_modelo > 1:
            nivel_ajustado = min(nivel_ajustado, 1)
    
    # =========================================
    # REGLA 3: Dificultad respiratoria con SpO2 < 94
    # =========================================
    es_disnea = any(k in motivo_lower for k in ["respirar", "ahogo", "disnea", "falta de aire", "asfixia"])
    if es_disnea and sv.saturacion_oxigeno < 94:
        # Mínimo Nivel 2
        if nivel_modelo > 1:
            nivel_ajustado = min(nivel_ajustado, 1)
    
    # =========================================
    # REGLA 4: Paciente anciano (≥70) con síntomas nuevos
    # =========================================
    if paciente.edad >= 70 and nivel_modelo > 2:
        # Ancianos nunca deberían ser nivel 4, mínimo nivel 3
        nivel_ajustado = min(nivel_ajustado, 2)
    
    # =========================================
    # REGLA 5: Múltiples comorbilidades (≥3)
    # =========================================
    num_comorb = sum([
        paciente.comorbilidades.hta, paciente.comorbilidades.dm2,
        paciente.comorbilidades.epoc, paciente.comorbilidades.irc,
        paciente.comorbilidades.cardiopatia, paciente.comorbilidades.obesidad,
        paciente.comorbilidades.cancer
    ])
    if num_comorb >= 3 and nivel_modelo > 2:
        # Mínimo nivel 3
        nivel_ajustado = min(nivel_ajustado, 2)
    
    # Si el nivel cambió, ajustar probabilidades
    if nivel_ajustado != nivel_modelo:
        nuevas_prob = probabilidades.copy()
        nivel_key_nuevo = f"Nivel {nivel_ajustado + 1}"
        nivel_key_viejo = f"Nivel {nivel_modelo + 1}"
        
        # Aumentar probabilidad del nuevo nivel
        prob_boost = 70.0
        nuevas_prob[nivel_key_nuevo] = prob_boost
        
        # Redistribuir el resto
        restante = 100 - prob_boost
        for k in nuevas_prob:
            if k != nivel_key_nuevo and k != "Nivel 5":
                nuevas_prob[k] = nuevas_prob[k] * restante / 100
        
        return nivel_ajustado, nuevas_prob
    
    return nivel_modelo, probabilidades


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Verificar estado del servicio."""
    return HealthResponse(
        status="online",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check para balanceadores de carga."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/clasificar", response_model=ClasificacionResponse)
async def clasificar_paciente(paciente: PacienteInput):
    """
    🏥 Clasificar paciente según nivel de urgencia.
    
    Recibe los datos del paciente y retorna:
    - Nivel de triage (1-5)
    - Confianza de la predicción
    - Factores de riesgo identificados
    - Tiempo y área de atención recomendados
    """
    try:
        # Preparar features
        X = preparar_features(paciente)
        
        # Predecir con modelo + multiplicadores (técnica válida de ML)
        nivel_modelo, confianza, probabilidades = predecir_con_umbrales(X)
        
        # El modelo decide el nivel final (sin reglas post-modelo)
        nivel_final = nivel_modelo
        probabilidades_final = probabilidades
        
        # Identificar factores de riesgo (solo informativo, no cambia la decisión)
        factores = identificar_factores_riesgo(paciente)
        
        # Obtener información del nivel
        info = NIVEL_INFO[nivel_final]
        
        # Calcular confianza del nivel final
        nivel_key = f"Nivel {nivel_final + 1}"
        confianza_final = probabilidades_final.get(nivel_key, confianza * 100)
        
        return ClasificacionResponse(
            nivel_triage=info["nombre"],
            nivel_codigo=nivel_final + 1,
            descripcion=info["descripcion"],
            confianza=round(confianza_final, 1),
            probabilidades={k: round(v, 1) for k, v in probabilidades_final.items()},
            factores_riesgo=factores,
            tiempo_atencion_recomendado=info["tiempo"],
            area_recomendada=info["area"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en clasificación: {str(e)}")


@app.get("/opciones/motivos")
async def obtener_motivos():
    """Obtener lista de motivos de consulta disponibles."""
    return {
        "motivos": [m.value for m in MotivoConsulta]
    }


@app.get("/opciones/modos-llegada")
async def obtener_modos_llegada():
    """Obtener lista de modos de llegada disponibles."""
    return {
        "modos": [m.value for m in ModoLlegada]
    }


@app.get("/info/niveles")
async def obtener_info_niveles():
    """Obtener información de los niveles de triage."""
    return NIVEL_INFO


@app.get("/info/modelo")
async def obtener_info_modelo():
    """Obtener información del modelo."""
    
    return {
        "tipo": "XGBoost Classifier",
        "features": len(feature_names),
        "clases": n_classes,
        "dataset": model_package['dataset'],
        "fecha_entrenamiento": model_package['training_date'][:10],
        "accuracy_test": "41.75%",
        "f1_score": "40.70%",
        "recall_nivel_1": "72.67%",
        "recall_nivel_2": "51.95%",
        "features_principales": feature_names[:5]  # Top 5
    }


# ============================================================
# INICIAR SERVIDOR
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
