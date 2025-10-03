# 🪐 Exoplanet Detector - NASA Space Apps Challenge

Detector de exoplanetas usando Machine Learning basado en datos del catálogo de Kepler.

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Linux/Mac con fish)
source venv/bin/activate.fish

# O para bash/zsh
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

```bash
# Forma simple - Todo automático
python run.py runstreamlit
```

¡Eso es todo! El script verificará automáticamente si necesita preprocesar datos o entrenar el modelo antes de iniciar la aplicación.

La aplicación estará disponible en: **http://localhost:8501**

## 📋 Comandos Disponibles

Si quieres ejecutar los pasos manualmente:

```bash
# Preprocesar datos raw
python run.py preprocess

# Generar características
python run.py features

# Entrenar modelo
python run.py train

# Iniciar aplicación web (con verificación automática)
python run.py runstreamlit
```

## 📁 Estructura del Proyecto

```
nasaspaceappschallenge/
├── data/
│   ├── raw/              # Datos originales de Kepler
│   └── processed/        # Datos procesados y características
├── models/               # Modelos entrenados
├── notebooks/            # Notebooks de análisis exploratorio
├── src/                  # Código fuente
│   ├── preprocessing.py  # Limpieza de datos
│   ├── feature_engineering.py  # Ingeniería de características
│   ├── model.py          # Entrenamiento del modelo
│   └── predict.py        # Predicciones
├── webapp/               # Aplicación Streamlit
│   └── app.py            # App principal
├── requirements.txt      # Dependencias
└── run.py               # Script de ejecución
```

## 🎯 Características

### 🚀 Funcionalidades Principales
- 🔍 **Detección automática de exoplanetas**
- 📊 **Clasificación en 3 categorías**: Confirmados, Candidatos, Falsos Positivos
- 🎨 **Interfaz web interactiva** con Streamlit
- 📈 **Precisión del modelo**: ~94-95%
- 📥 **Subida y procesamiento automático** de archivos CSV
- 💾 **Descarga de predicciones** en formato CSV

### 📊 Visualizaciones Incluidas
- 📈 **Distribución de predicciones** - Gráfico de barras con conteo por categoría
- 🎯 **Confianza del modelo** - Histograma de probabilidades máximas
- 🔝 **Top 15 características importantes** - Gráfico de importancia de features
- 📉 **Matriz de correlación** - Heatmap de correlación entre features principales
- 🎯 **Matriz de confusión** - Evaluación del rendimiento (si hay etiquetas)
- 📋 **Reporte de clasificación** - Métricas detalladas (precision, recall, f1-score)

### ⚡ Procesamiento Automático
Al subir un CSV, la aplicación automáticamente:
1. ✅ Detecta y limpia los datos
2. ✅ Aplica preprocesamiento (imputación, normalización)
3. ✅ Genera características relevantes
4. ✅ Realiza predicciones con el modelo entrenado
5. ✅ Genera visualizaciones interactivas

## 🛠️ Tecnologías

- **Python 3.13**
- **LightGBM** - Modelo de clasificación
- **Streamlit** - Interfaz web
- **Pandas & NumPy** - Procesamiento de datos
- **Scikit-learn** - Pipeline de ML
- **Matplotlib & Seaborn** - Visualizaciones
- **Astropy** - Cálculos astronómicos

## 🧪 Archivo de Prueba

Para probar rápidamente la aplicación, usa el archivo `sample_data.csv` incluido en el proyecto (100 registros de ejemplo).

