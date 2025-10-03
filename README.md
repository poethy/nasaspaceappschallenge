# 🪐 Exoplanet Detector - NASA Space Apps Challenge 2025

Detector de exoplanetas usando Machine Learning con modelos especializados para las misiones **KEPLER**, **TESS** y **K2**.

**Desarrollado por:** [Equipo NoLit Developers](https://github.com/NoLit-Developers)

---

## 🚀 Inicio Rápido - Guía Paso a Paso

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/poethy/nasaspaceappschallenge.git
cd nasaspaceappschallenge
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv
```

### Paso 3: Activar Entorno Virtual

**En Linux/Mac (Fish Shell):**
```bash
source venv/bin/activate.fish
```

**En Linux/Mac (Bash/Zsh):**
```bash
source venv/bin/activate
```

**En Windows:**
```bash
venv\Scripts\activate
```

### Paso 4: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 5: Ejecutar la Aplicación

```bash
python run.py runstreamlit
```

¡Eso es todo! El script verificará automáticamente si necesita preprocesar datos o entrenar los modelos antes de iniciar la aplicación.

**La aplicación estará disponible en:** http://localhost:8501

---

## 🎯 Modelos y Precisión

El proyecto incluye **3 modelos especializados** entrenados con LightGBM:

| Misión | Dataset | Precisión | Objetos Entrenados |
|--------|---------|-----------|-------------------|
| **KEPLER** | cumulativeKEPLER.csv | **94%** | ~9,500 objetos |
| **TESS (TOI)** | cumulativeTOI.csv | **77%** | ~7,700 objetos |
| **K2** | cumulativeK2.csv | **97%** | Objetos confirmados |

Cada modelo está optimizado para las características específicas de su misión.

---

## 📋 Comandos Disponibles

### Entrenar Todos los Modelos

```bash
# Entrenar los 3 modelos (KEPLER, TESS, K2)
python run.py train_all
```

### Entrenar Modelos Individuales

```bash
# Solo KEPLER
python run.py train_kepler

# Solo TESS
python run.py train_tess

# Solo K2
python run.py train_k2
```

### Comandos de Procesamiento Manual

Si prefieres ejecutar los pasos manualmente:

```bash
# Preprocesar datos raw
python run.py preprocess

# Generar características
python run.py features

# Entrenar modelo (usa KEPLER por defecto)
python run.py train

# Iniciar aplicación web
python run.py runstreamlit
```

---

## 📁 Estructura del Proyecto

```
nasaspaceappschallenge/
├── data/
│   ├── raw/                      # Datos originales
│   │   ├── cumulativeKEPLER.csv  # Dataset KEPLER
│   │   ├── cumulativeTOI.csv     # Dataset TESS
│   │   └── cumulativeK2.csv      # Dataset K2
│   └── processed/                # Datos procesados
│       ├── kepler_clean.csv
│       ├── kepler_features.csv
│       ├── tess_clean.csv
│       ├── tess_features.csv
│       ├── k2_clean.csv
│       └── k2_features.csv
├── models/                       # Modelos entrenados
│   ├── exoplanet_model_kepler.joblib
│   ├── exoplanet_model_kepler.scaler.joblib
│   ├── exoplanet_model_kepler.features.joblib
│   ├── exoplanet_model_tess.joblib
│   ├── exoplanet_model_tess.scaler.joblib
│   ├── exoplanet_model_tess.features.joblib
│   ├── exoplanet_model_k2.joblib
│   ├── exoplanet_model_k2.scaler.joblib
│   └── exoplanet_model_k2.features.joblib
├── src/                          # Código fuente
│   ├── preprocessing.py          # Limpieza y preprocesamiento
│   ├── feature_engineering.py    # Generación de características
│   ├── model.py                  # Entrenamiento del modelo
│   └── predict.py                # Predicciones
├── webapp/                       # Aplicación Streamlit
│   └── app.py                    # Aplicación web principal
├── requirements.txt              # Dependencias del proyecto
├── run.py                        # Script de ejecución principal
└── README.md                     # Este archivo
```

---

## 🎯 Características de la Aplicación

### 🚀 Funcionalidades Principales

- 🔍 **Detección automática de exoplanetas**
- 🌌 **Soporte multi-misión**: KEPLER, TESS (TOI) y K2
- 📊 **Clasificación en 3 categorías**:
  - ✅ **Confirmados**: Exoplanetas verificados
  - 🔍 **Candidatos**: Probables exoplanetas que requieren confirmación
  - ❌ **Falsos Positivos**: Señales que no son exoplanetas
- 🎨 **Interfaz web interactiva** con Streamlit
- 📥 **Subida y procesamiento automático** de archivos CSV
- 💾 **Descarga de predicciones** en formato CSV
- 🎯 **Top 10 exoplanetas** ordenados por confianza

### 📊 Visualizaciones Incluidas

- 📊 **Estadísticas del Dataset** - Métricas clave del archivo subido
- 📈 **Distribución de Predicciones** - Conteo por categoría (Confirmado/Candidato/Falso Positivo)
- 🎯 **Distribución de Confianza** - Histograma de probabilidades máximas
- 🔝 **Top 15 Características Importantes** - Features más influyentes del modelo
- 🔥 **Matriz de Correlación** - Heatmap de las principales features
- 📉 **Matriz de Confusión** - Evaluación de rendimiento (si hay etiquetas reales)
- 📋 **Reporte de Clasificación** - Precision, Recall, F1-Score por clase
- 🌟 **Exoplanetas Descubiertos** - Lista detallada con métricas y confianza

### ⚡ Procesamiento Automático

Al subir un CSV, la aplicación automáticamente:

1. ✅ **Detecta el formato** y extrae la tabla de datos
2. ✅ **Limpia y normaliza** los valores numéricos
3. ✅ **Aplica preprocesamiento** (imputación de valores faltantes, transformaciones logarítmicas)
4. ✅ **Genera características derivadas** (ratios, interacciones, features estadísticas)
5. ✅ **Carga el modelo especializado** según la misión seleccionada
6. ✅ **Realiza predicciones** con probabilidades de cada clase
7. ✅ **Genera visualizaciones interactivas** y análisis detallado

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.13**
- **LightGBM** - Gradient Boosting para clasificación
- **Streamlit** - Framework de interfaz web
- **Pandas & NumPy** - Procesamiento y análisis de datos
- **Scikit-learn** - Pipeline de Machine Learning
- **Matplotlib & Seaborn** - Visualizaciones científicas
- **Joblib** - Serialización de modelos
- **Astropy** - Cálculos astronómicos (opcional)

---

## 📊 Uso de la Aplicación

### 1. Seleccionar Misión

En el sidebar, elige entre:
- **KEPLER** - Misión original de búsqueda de exoplanetas (2009-2018)
- **TESS (TOI)** - Transiting Exoplanet Survey Satellite
- **K2** - Extensión de la misión Kepler

### 2. Subir Dataset

Arrastra o selecciona un archivo CSV del catálogo correspondiente. La aplicación acepta:
- `cumulativeKEPLER.csv` para KEPLER
- `cumulativeTOI.csv` para TESS
- `cumulativeK2.csv` para K2

### 3. Analizar Resultados

La aplicación mostrará automáticamente:
- **Estadísticas generales** del dataset
- **Predicciones del modelo** con probabilidades
- **Gráficas de análisis** del modelo
- **Lista de exoplanetas descubiertos** ordenados por confianza
- **Resultados detallados** para todos los objetos

### 4. Descargar Resultados

Descarga los resultados en formato CSV:
- **Exoplanetas filtrados** - Solo objetos clasificados como confirmados/candidatos
- **Resultados completos** - Todos los objetos con predicciones

---

## 🧪 Archivos de Prueba

Los datasets originales deben colocarse en `data/raw/`:
- `cumulativeKEPLER.csv` - Dataset de Kepler
- `cumulativeTOI.csv` - Dataset de TESS (TOI)
- `cumulativeK2.csv` - Dataset de K2

Estos archivos se pueden obtener del NASA Exoplanet Archive:
https://exoplanetarchive.ipac.caltech.edu/

---

## 🔬 Metodología

### Preprocesamiento
1. Detección automática del inicio de tabla en CSVs con metadatos
2. Conversión de columnas a formato numérico
3. Imputación de valores faltantes con la mediana
4. Transformaciones logarítmicas para variables con alta asimetría
5. Extracción y mapeo de etiquetas de disposición

### Feature Engineering
1. Creación de ratios (profundidad/radio, período/duración)
2. Transformaciones logarítmicas (`log1p`)
3. Features derivadas de parámetros orbitales
4. Normalización con StandardScaler

### Modelo
- **Algoritmo**: LightGBM (Light Gradient Boosting Machine)
- **Tipo**: Clasificación multiclase (3 clases)
- **Validación**: Cross-validation con GroupKFold (KEPLER) o StratifiedKFold (TESS/K2)
- **Hiperparámetros**:
  - `n_estimators`: 300
  - `learning_rate`: 0.05
  - `num_leaves`: 31
  - `objective`: 'multiclass'

---

## 📈 Resultados del Modelo

### KEPLER (94% precisión)
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1584  (Falso Positivo)
           1       0.89      0.85      0.87       657  (Candidato)
           2       0.90      0.93      0.92       947  (Confirmado)

    accuracy                           0.94      3188
```

### TESS (77% precisión)
Modelo optimizado para detecciones recientes de TESS con características diferentes.

### K2 (97% precisión)
Modelo con alta precisión para objetos confirmados de la misión K2.

---

## 🤝 Contribuciones

Este proyecto fue desarrollado para el **NASA Space Apps Challenge 2025** por el [**NoLit Developers**](https://github.com/NoLit-Developers).

---

## 📝 Licencia

Este proyecto está desarrollado con fines educativos para el NASA Space Apps Challenge 2025.

---

## 🌟 Agradecimientos

- **NASA Exoplanet Archive** - Por proporcionar los datasets
- **Misión Kepler/K2** - Por los datos de exoplanetas
- **Misión TESS** - Por las observaciones de tránsitos
- **Comunidad de Machine Learning** - Por las herramientas open source

---

**🚀 Desarrollado para NASA Space Apps Challenge 2025 | [Equipo NoLit Developers](https://github.com/NoLit-Developers)**
