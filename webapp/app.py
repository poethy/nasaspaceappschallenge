"""
Streamlit app para detección de exoplanetas:
- Subir CSV con datos de Kepler
- Preprocesar automáticamente
- Generar características
- Predecir con modelo entrenado
- Mostrar gráficas y análisis
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io
import subprocess

# Configurar matplotlib para español
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# Asegurarse que src esté en path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Importar funciones de preprocesamiento
from preprocessing import detect_table_start, parse_table_from, to_numeric_columns, impute_median
from feature_engineering import basic_feature_engineering

st.set_page_config(page_title="🪐 Exoplanet Detector", layout="wide", page_icon="🪐")

# Header dinámico según misión (se actualizará después del sidebar)
# Temporalmente vacío, se llenará después
mission_placeholder = st.empty()

# Sidebar con información y selector de misión
with st.sidebar:
    st.header("🚀 Selecciona la Misión")
    
    mission = st.radio(
        "¿Qué misión quieres analizar?",
        options=["KEPLER", "TESS (TOI)", "K2"],
        help="Cada misión tiene un modelo especializado entrenado con sus datos específicos"
    )
    
    st.markdown("---")
    
    # Información según la misión
    mission_info = {
        "KEPLER": {
            "name": "Kepler",
            "launched": "2009",
            "planets": "~2,800",
            "columns": "kepid, koi_*",
            "icon": "🔭"
        },
        "TESS (TOI)": {
            "name": "TESS",
            "launched": "2018",
            "planets": "~400+",
            "columns": "toi, tid, pl_*",
            "icon": "🛰️"
        },
        "K2": {
            "name": "K2",
            "launched": "2014",
            "planets": "~500",
            "columns": "pl_name, disposition",
            "icon": "🌌"
        }
    }
    
    info = mission_info[mission]
    st.header(f"{info['icon']} {info['name']}")
    st.markdown(f"""
    **Lanzamiento:** {info['launched']}  
    **Exoplanetas:** {info['planets']}  
    **Columnas clave:** `{info['columns']}`
    """)
    
    st.markdown("---")
    st.header("ℹ️ Instrucciones")
    st.markdown("""
    1. Selecciona la misión arriba
    2. Sube el CSV correspondiente
    3. El sistema procesará automáticamente
    4. Predice: **Confirmado**, **Candidato** o **Falso Positivo**
    
    **Modelo:** LightGBM  
    **Precisión:** ~94-95%
    """)
    
    st.markdown("---")
    st.markdown("**NASA Space Apps Challenge 2025 | Equipo: NoLit Developers**")

# Actualizar header según misión seleccionada
with mission_placeholder.container():
    st.title(f"{info['icon']} Detector de Exoplanetas - {info['name']}")
    st.markdown(f"### Clasificador especializado para la misión {info['name']}")
    st.markdown("---")

# Configuración según misión
mission_config = {
    "KEPLER": {
        "model_path": "models/exoplanet_model_kepler.joblib",
        "dataset_example": "cumulativeKEPLER.csv",
        "id_col": "kepid",
        "name_col": "kepoi_name"
    },
    "TESS (TOI)": {
        "model_path": "models/exoplanet_model_tess.joblib",
        "dataset_example": "cumulativeTOI.csv",
        "id_col": "tid",
        "name_col": "toi"
    },
    "K2": {
        "model_path": "models/exoplanet_model_k2.joblib",
        "dataset_example": "cumulativeK2.csv",
        "id_col": "hostname",
        "name_col": "pl_name"
    }
}

config = mission_config[mission]

# Upload section
st.header("📤 1. Subir Dataset")
uploaded = st.file_uploader(
    f"Arrastra o selecciona un archivo CSV de {info['name']}",
    type=['csv'],
    help=f"El archivo debe ser del catálogo {info['name']}. Ejemplo: {config['dataset_example']}"
)

if not uploaded:
    st.info("👆 Por favor, sube un archivo CSV para comenzar el análisis.")
    st.stop()

# Procesamiento automático
with st.spinner("🔄 Procesando datos..."):
    try:
        # Detectar inicio de tabla
        temp_file = f"/tmp/uploaded_{uploaded.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded.getbuffer())
        
        start_idx = detect_table_start(temp_file)
        if start_idx is not None:
            df_pd, df_raw = parse_table_from(temp_file, start_idx)
            df = to_numeric_columns(df_pd)
        else:
            df = pd.read_csv(uploaded)
        
        # Aplicar preprocesamiento
        df = impute_median(df)
        
        # Feature engineering
        df = basic_feature_engineering(df)
        
        st.success("✅ Datos procesados exitosamente!")
        
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        st.stop()

# Mostrar estadísticas del dataset
st.header("📊 2. Estadísticas del Dataset")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Objetos", len(df))
with col2:
    st.metric("Características", len(df.columns))
with col3:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    st.metric("Columnas Numéricas", len(numeric_cols))
with col4:
    if 'target_label' in df.columns:
        st.metric("Etiquetas Únicas", df['target_label'].nunique())
    else:
        st.metric("Valores Nulos", df.isnull().sum().sum())

# Preview de datos
with st.expander("👁️ Ver muestra de datos (primeras 10 filas)"):
    st.dataframe(df.head(10), use_container_width=True)

# Verificar modelo según misión
model_path = Path(config["model_path"])
if not model_path.exists():
    st.error(f"❌ No se encontró modelo para {info['name']}. Entrenando modelo automáticamente...")
    
    # Entrenar modelo automáticamente
    with st.spinner(f"🤖 Entrenando modelo especializado para {info['name']}... Esto puede tomar unos minutos."):
        try:
            mission_key = mission.replace(" (TOI)", "").lower()
            result = subprocess.run(
                ["python", "run.py", "train_all"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                st.success(f"✅ Modelo de {info['name']} entrenado exitosamente!")
                st.rerun()
            else:
                st.error(f"Error al entrenar modelo: {result.stderr}")
                st.stop()
        except Exception as e:
            st.error(f"Error al entrenar modelo: {str(e)}")
            st.stop()

# Cargar modelo
@st.cache_resource
def load_model(model_file):
    model = joblib.load(str(model_file))
    scaler = joblib.load(str(model_file).replace('.joblib', '.scaler.joblib'))
    features = joblib.load(str(model_file).replace('.joblib', '.features.joblib'))
    return model, scaler, features

model, scaler, features = load_model(model_path)

# Predicciones
st.header("🔮 3. Predicciones del Modelo")

with st.spinner("🤖 Generando predicciones..."):
    # Preparar datos
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    X = df[numeric_cols].copy()
    
    # Guardar labels si existen
    has_labels = 'target_label' in X.columns
    if has_labels:
        y_true = X['target_label'].values
        X = X.drop(columns=['target_label'])
    
    # Preparar matriz de features
    X_model = pd.DataFrame(0, index=X.index, columns=features)
    for c in X.columns:
        if c in X_model.columns:
            X_model[c] = X[c].values
    
    # Escalar y predecir
    X_scaled = scaler.transform(X_model.values)
    probs = model.predict_proba(X_scaled)
    predictions = model.predict(X_scaled)
    
    # Crear DataFrame de resultados
    labels = {0: 'Falso Positivo', 1: 'Candidato', 2: 'Confirmado'}
    df_predictions = pd.DataFrame({
        'Predicción': [labels[p] for p in predictions],
        'Prob. Falso Positivo': probs[:, 0],
        'Prob. Candidato': probs[:, 1],
        'Prob. Confirmado': probs[:, 2]
    })

st.success(f"✅ {len(predictions)} predicciones generadas!")

# Análisis de descubrimientos
num_confirmados = (predictions == 2).sum()
num_candidatos = (predictions == 1).sum()
num_exoplanetas = num_confirmados + num_candidatos
num_falsos = (predictions == 0).sum()

# Métricas destacadas de descubrimientos
st.markdown("### 🌟 Descubrimientos de Exoplanetas")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🪐 Exoplanetas Totales", 
        num_exoplanetas,
        delta=f"{(num_exoplanetas/len(predictions)*100):.1f}%",
        help="Candidatos + Confirmados"
    )
with col2:
    st.metric(
        "✅ Confirmados", 
        num_confirmados,
        delta=f"{(num_confirmados/len(predictions)*100):.1f}%"
    )
with col3:
    st.metric(
        "🔍 Candidatos", 
        num_candidatos,
        delta=f"{(num_candidatos/len(predictions)*100):.1f}%"
    )
with col4:
    st.metric(
        "❌ Falsos Positivos", 
        num_falsos,
        delta=f"-{(num_falsos/len(predictions)*100):.1f}%",
        delta_color="inverse"
    )

if num_exoplanetas > 0:
    st.success(f"🎉 **¡El modelo ha identificado {num_exoplanetas} posibles exoplanetas en tu dataset!**")
    
    # Filtrar exoplanetas de alta confianza
    umbral_confianza = 0.7
    exoplanetas_alta_confianza = (
        ((predictions == 2) & (probs[:, 2] >= umbral_confianza)) |
        ((predictions == 1) & (probs[:, 1] >= umbral_confianza))
    )
    num_alta_confianza = exoplanetas_alta_confianza.sum()
    
    if num_alta_confianza > 0:
        st.info(f"🌟 **{num_alta_confianza} exoplanetas tienen confianza ≥ {umbral_confianza*100:.0f}%** - ¡Estos son los descubrimientos más prometedores!")
else:
    st.warning("⚠️ No se identificaron exoplanetas candidatos o confirmados en este dataset.")

st.markdown("---")

# Mostrar distribución de predicciones
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Distribución de Predicciones")
    fig, ax = plt.subplots(figsize=(8, 6))
    pred_counts = pd.Series(predictions).map(labels).value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax.bar(pred_counts.index, pred_counts.values, color=colors)
    ax.set_xlabel('Clasificación', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cantidad', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Clasificaciones', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15)
    for i, v in enumerate(pred_counts.values):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("🎯 Distribución de Probabilidades")
    fig, ax = plt.subplots(figsize=(8, 6))
    max_probs = probs.max(axis=1)
    ax.hist(max_probs, bins=30, color='#45B7D1', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Probabilidad Máxima', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Confianza de las Predicciones', fontsize=14, fontweight='bold')
    ax.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {max_probs.mean():.3f}')
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Importancia de características
st.header("📊 4. Análisis del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔝 Top 15 Características Importantes")
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Característica': features,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(importance_df)), importance_df['Importancia'], color='#FF6B6B')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Característica'], fontsize=9)
        ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
        ax.set_title('Características Más Relevantes', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("El modelo no proporciona importancia de características")

with col2:
    st.subheader("📉 Matriz de Correlación (Top Features)")
    if hasattr(model, 'feature_importances_'):
        top_features = importance_df['Característica'].head(10).tolist()
        correlation_data = X_model[top_features].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax, cbar_kws={'label': 'Correlación'})
        ax.set_title('Correlación entre Top Features', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig)
        plt.close()

# Matriz de confusión si hay labels
if has_labels:
    st.subheader("🎯 Matriz de Confusión")
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Filtrar solo las filas con labels válidos
    valid_idx = (y_true >= 0) & (y_true <= 2)
    y_true_valid = y_true[valid_idx]
    y_pred_valid = predictions[valid_idx]
    
    if len(y_true_valid) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(y_true_valid, y_pred_valid)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=list(labels.values()),
                        yticklabels=list(labels.values()),
                        ax=ax)
            ax.set_ylabel('Real', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
            ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("📋 Reporte de Clasificación")
            report = classification_report(y_true_valid, y_pred_valid, 
                                          target_names=list(labels.values()),
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# Tabla de exoplanetas descubiertos
st.header("🌟 5. Exoplanetas Descubiertos")

if num_exoplanetas > 0:
    # Filtrar solo exoplanetas (candidatos + confirmados)
    mask_exoplanetas = (predictions == 1) | (predictions == 2)
    
    # Crear DataFrame de exoplanetas
    df_exoplanetas = df.reset_index(drop=True)[mask_exoplanetas].copy()
    df_pred_exoplanetas = df_predictions[mask_exoplanetas].copy()
    
    # Columnas a mostrar según la misión
    if mission == "KEPLER":
        columnas_exoplanetas = [
            'kepid', 'kepoi_name', 'kepler_name',
            'Predicción', 'Prob. Confirmado', 'Prob. Candidato',
            'koi_period', 'koi_depth', 'koi_duration',
            'koi_prad', 'koi_teq', 'koi_insol',
            'koi_srad', 'koi_steff',
            'koi_model_snr', 'koi_score'
        ]
    elif mission == "TESS (TOI)":
        columnas_exoplanetas = [
            'tid', 'toi',
            'Predicción', 'Prob. Confirmado', 'Prob. Candidato',
            'pl_orbper', 'pl_trandep', 'pl_trandur',
            'pl_rade', 'pl_eqt', 'pl_insol',
            'st_rad', 'st_teff',
            'pl_dens', 'tfopwg_disp'
        ]
    else:  # K2
        columnas_exoplanetas = [
            'hostname', 'pl_name',
            'Predicción', 'Prob. Confirmado', 'Prob. Candidato',
            'pl_orbper', 'pl_trandep', 'pl_trandur',
            'pl_rade', 'pl_eqt', 'pl_insol',
            'st_rad', 'st_teff',
            'pl_dens', 'disposition'
        ]
    
    # Combinar datos
    exoplanetas_completo = pd.concat([
        df_exoplanetas.reset_index(drop=True),
        df_pred_exoplanetas.reset_index(drop=True)
    ], axis=1)
    
    # Filtrar columnas disponibles
    cols_disponibles = [col for col in columnas_exoplanetas if col in exoplanetas_completo.columns]
    
    # Asegurar que las columnas de identificación estén incluidas
    id_cols_needed = []
    if mission == "KEPLER":
        id_cols_needed = ['kepid', 'kepoi_name', 'kepler_name']
    elif mission == "TESS (TOI)":
        id_cols_needed = ['tid', 'toi']
    else:  # K2
        id_cols_needed = ['hostname', 'pl_name']
    
    # Agregar columnas de ID que existan y no estén ya en la lista
    for id_col in id_cols_needed:
        if id_col in exoplanetas_completo.columns and id_col not in cols_disponibles:
            cols_disponibles.insert(0, id_col)
    
    if not cols_disponibles:
        cols_disponibles = list(exoplanetas_completo.columns[:15])
    
    exoplanetas_mostrar = exoplanetas_completo[cols_disponibles].copy()
    
    # Ordenar por probabilidad de confirmado (descendente)
    if 'Prob. Confirmado' in exoplanetas_mostrar.columns:
        exoplanetas_mostrar = exoplanetas_mostrar.sort_values('Prob. Confirmado', ascending=False)
    
    st.markdown(f"""
    ### 📊 Lista de {len(exoplanetas_mostrar)} Exoplanetas Identificados
    
    **Interpretación:**
    - ✅ **Confirmado**: El modelo tiene alta confianza de que es un exoplaneta real
    - 🔍 **Candidato**: Probable exoplaneta que requiere confirmación adicional
    - 🎯 **Probabilidad**: Mayor probabilidad = Mayor confianza del modelo
    """)
    
    # Tabs para diferentes vistas
    tab1, tab2 = st.tabs(["📋 Tabla Completa", "🏆 Top 10 Confianza"])
    
    with tab1:
        st.dataframe(
            exoplanetas_mostrar.style.format({
                'Prob. Confirmado': '{:.3f}',
                'Prob. Candidato': '{:.3f}',
            }, na_rep='-'),
            width='stretch',
            height=400
        )
    
    with tab2:
        st.markdown("### 🏆 Top 10 Exoplanetas con Mayor Confianza")
        top10 = exoplanetas_mostrar.head(10)
        
        for rank, (idx, row) in enumerate(top10.iterrows(), start=1):
            # Identificador dinámico según misión
            if mission == "KEPLER":
                # Intentar diferentes columnas de identificación
                if 'kepoi_name' in row and pd.notna(row['kepoi_name']) and row['kepoi_name'] != '':
                    obj_id = row['kepoi_name']
                elif 'kepid' in row and pd.notna(row['kepid']):
                    obj_id = f"KepID-{int(row['kepid'])}"
                elif 'kepler_name' in row and pd.notna(row['kepler_name']) and row['kepler_name'] != '':
                    obj_id = row['kepler_name']
                else:
                    obj_id = f"Exoplaneta {rank}"
            elif mission == "TESS (TOI)":
                if 'toi' in row and pd.notna(row['toi']) and row['toi'] != '':
                    obj_id = f"TOI-{row['toi']}"
                elif 'tid' in row and pd.notna(row['tid']):
                    obj_id = f"TIC-{int(row['tid'])}"
                else:
                    obj_id = f"Exoplaneta {rank}"
            else:  # K2
                if 'pl_name' in row and pd.notna(row['pl_name']) and row['pl_name'] != '':
                    obj_id = row['pl_name']
                elif 'hostname' in row and pd.notna(row['hostname']) and row['hostname'] != '':
                    obj_id = row['hostname']
                else:
                    obj_id = f"Exoplaneta {rank}"
            
            with st.expander(f"🪐 #{rank} - {obj_id} - {row['Predicción']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Confianza", f"{row['Prob. Confirmado']*100:.1f}%")
                    # Período orbital (diferentes columnas por misión)
                    period_col = 'koi_period' if mission == "KEPLER" else 'pl_orbper'
                    if period_col in row and pd.notna(row[period_col]):
                        st.metric("Período Orbital", f"{row[period_col]:.2f} días")
                
                with col2:
                    # Radio del planeta
                    rad_col = 'koi_prad' if mission == "KEPLER" else 'pl_rade'
                    if rad_col in row and pd.notna(row[rad_col]):
                        st.metric("Radio Planeta", f"{row[rad_col]:.2f} R⊕")
                    # Temperatura
                    temp_col = 'koi_teq' if mission == "KEPLER" else 'pl_eqt'
                    if temp_col in row and pd.notna(row[temp_col]):
                        st.metric("Temperatura", f"{row[temp_col]:.0f} K")
                
                with col3:
                    # Profundidad del tránsito
                    depth_col = 'koi_depth' if mission == "KEPLER" else 'pl_trandep'
                    if depth_col in row and pd.notna(row[depth_col]):
                        st.metric("Profundidad Tránsito", f"{row[depth_col]:.2f} ppm")
                    # Radio estelar
                    srad_col = 'koi_srad' if mission == "KEPLER" else 'st_rad'
                    if srad_col in row and pd.notna(row[srad_col]):
                        st.metric("Radio Estelar", f"{row[srad_col]:.2f} R☉")
    
    # Botón de descarga de exoplanetas
    st.markdown("---")
    csv_exoplanetas = exoplanetas_mostrar.to_csv(index=False)
    st.download_button(
        label="📥 Descargar solo exoplanetas descubiertos (CSV)",
        data=csv_exoplanetas,
        file_name="exoplanetas_descubiertos.csv",
        mime="text/csv",
        use_container_width=True,
        help=f"Descarga solo los {len(exoplanetas_mostrar)} exoplanetas identificados"
    )
else:
    st.info("No se identificaron exoplanetas en este dataset. Todos los objetos fueron clasificados como falsos positivos.")

st.markdown("---")

# Tabla de resultados completos
st.header("📋 6. Resultados Detallados (Todos los Objetos)")

# Crear DataFrame completo con todas las columnas
result_df_complete = pd.concat([
    df.reset_index(drop=True),
    df_predictions.reset_index(drop=True)
], axis=1)

# Definir columnas relevantes para el usuario
columnas_importantes = [
    # Identificación
    'kepid', 'kepoi_name', 'kepler_name',
    # Predicciones (siempre al inicio)
    'Predicción', 'Prob. Confirmado', 'Prob. Candidato', 'Prob. Falso Positivo',
    # Características orbitales
    'koi_period', 'koi_duration', 'koi_time0bk',
    # Características de tránsito (picos de luz)
    'koi_depth', 'koi_ror', 'koi_impact',
    # Características del planeta
    'koi_prad', 'koi_teq', 'koi_insol',
    # Características de la estrella
    'koi_srad', 'koi_steff', 'koi_slogg',
    # SNR y calidad de datos
    'koi_model_snr', 'koi_tce_plnt_num',
    # Disposición original si existe
    'koi_disposition', 'koi_pdisposition', 'koi_score'
]

# Filtrar solo las columnas que existen en el DataFrame
columnas_disponibles = [col for col in columnas_importantes if col in result_df_complete.columns]

# Si tenemos predicciones, moverlas al inicio
if 'Predicción' in columnas_disponibles:
    pred_cols = ['Predicción', 'Prob. Confirmado', 'Prob. Candidato', 'Prob. Falso Positivo']
    pred_cols = [col for col in pred_cols if col in columnas_disponibles]
    otras_cols = [col for col in columnas_disponibles if col not in pred_cols]
    columnas_disponibles = pred_cols + otras_cols

# Crear DataFrame filtrado
if columnas_disponibles:
    result_df_display = result_df_complete[columnas_disponibles].copy()
else:
    # Si no encontramos columnas específicas, mostrar las numéricas más relevantes
    numeric_cols = result_df_complete.select_dtypes(include=[np.number]).columns[:15].tolist()
    pred_cols = ['Predicción', 'Prob. Confirmado', 'Prob. Candidato', 'Prob. Falso Positivo']
    display_cols = [col for col in pred_cols if col in result_df_complete.columns] + numeric_cols
    result_df_display = result_df_complete[display_cols]

st.markdown("""
**Columnas mostradas:**
- 🎯 **Predicciones**: Clasificación y probabilidades del modelo
- 🌍 **Características del planeta**: Radio (koi_prad), temperatura (koi_teq)
- 🔆 **Tránsito de luz**: Profundidad (koi_depth), duración del tránsito
- ⭐ **Características estelares**: Radio estelar, temperatura efectiva
- 📊 **Métricas de calidad**: SNR, score de confianza
""")

# Mostrar con formato mejorado
st.dataframe(
    result_df_display.head(50).style.format({
        'Prob. Confirmado': '{:.3f}',
        'Prob. Candidato': '{:.3f}',
        'Prob. Falso Positivo': '{:.3f}',
    }, na_rep='-'),
    use_container_width=True,
    height=400
)

# Descarga de resultados
st.header("💾 7. Descargar Resultados")

col1, col2 = st.columns(2)

with col1:
    # Descargar solo columnas relevantes
    csv_filtrado = result_df_display.to_csv(index=False)
    st.download_button(
        label="📥 Descargar resultados filtrados (CSV)",
        data=csv_filtrado,
        file_name="exoplanet_predictions_filtrado.csv",
        mime="text/csv",
        use_container_width=True,
        help="Descarga solo las columnas más relevantes mostradas en la tabla"
    )

with col2:
    # Descargar todas las columnas
    csv_completo = result_df_complete.to_csv(index=False)
    st.download_button(
        label="📥 Descargar resultados completos (CSV)",
        data=csv_completo,
        file_name="exoplanet_predictions_completo.csv",
        mime="text/csv",
        use_container_width=True,
        help="Descarga todas las columnas incluyendo características técnicas"
    )

# Footer
st.markdown("---")
st.markdown("**🚀 Desarrollado para NASA Space Apps Challenge 2025 | Equipo: NoLit Developers**")
st.markdown("---")
st.link_button("🔙 Volver al Portal Principal", "https://06067336e270.ngrok-free.app/")
