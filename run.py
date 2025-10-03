"""
Script simple para ejecutar tareas comunes.
Ejemplo de uso:
    python run.py preprocess
    python run.py train
    python run.py runstreamlit
"""
import sys, subprocess
from pathlib import Path

def train_mission(mission_name, raw_file, clean_file, features_file, model_file):
    """Entrena un modelo para una misión específica"""
    print(f"\n{'='*60}")
    print(f"🚀 Entrenando modelo para {mission_name}")
    print(f"{'='*60}\n")
    
    # Preprocesar
    print(f"⚙️  Preprocesando {mission_name}...")
    subprocess.run(["python","src/preprocessing.py", raw_file, clean_file])
    
    # Feature engineering
    print(f"⚙️  Generando características...")
    subprocess.run(["python","src/feature_engineering.py", clean_file, features_file])
    
    # Entrenar
    print(f"🤖 Entrenando modelo...")
    subprocess.run(["python","src/model.py", features_file, model_file])
    
    print(f"\n✅ Modelo de {mission_name} completado!\n")

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if cmd == "train_all":
        # Entrenar los 3 modelos
        missions = [
            ("KEPLER", "data/raw/cumulativeKEPLER.csv", "data/processed/kepler_clean.csv", 
             "data/processed/kepler_features.csv", "models/exoplanet_model_kepler.joblib"),
            ("TESS", "data/raw/cumulativeTOI.csv", "data/processed/tess_clean.csv",
             "data/processed/tess_features.csv", "models/exoplanet_model_tess.joblib"),
            ("K2", "data/raw/cumulativeK2.csv", "data/processed/k2_clean.csv",
             "data/processed/k2_features.csv", "models/exoplanet_model_k2.joblib")
        ]
        
        for mission_name, raw, clean, features, model in missions:
            train_mission(mission_name, raw, clean, features, model)
        
        print("\n🎉 ¡Todos los modelos entrenados exitosamente!\n")
    
    elif cmd == "train_kepler":
        train_mission("KEPLER", "data/raw/cumulativeKEPLER.csv", "data/processed/kepler_clean.csv",
                     "data/processed/kepler_features.csv", "models/exoplanet_model_kepler.joblib")
    
    elif cmd == "train_tess":
        train_mission("TESS", "data/raw/cumulativeTOI.csv", "data/processed/tess_clean.csv",
                     "data/processed/tess_features.csv", "models/exoplanet_model_tess.joblib")
    
    elif cmd == "train_k2":
        train_mission("K2", "data/raw/cumulativeK2.csv", "data/processed/k2_clean.csv",
                     "data/processed/k2_features.csv", "models/exoplanet_model_k2.joblib")
    
    elif cmd == "preprocess":
        subprocess.run(["python","src/preprocessing.py","data/raw/cumulativeKEPLER.csv","data/processed/exoplanets_clean.csv"])
    elif cmd == "features":
        subprocess.run(["python","src/feature_engineering.py","data/processed/exoplanets_clean.csv","data/processed/exoplanets_features.csv"])
    elif cmd == "train":
        subprocess.run(["python","src/model.py","data/processed/exoplanets_features.csv","models/exoplanet_model.joblib"])
    elif cmd == "runstreamlit":
        # Verificar si existen los archivos necesarios
        clean_data = Path("data/processed/exoplanets_clean.csv")
        features_data = Path("data/processed/exoplanets_features.csv")
        model_file = Path("models/exoplanet_model.joblib")
        
        # Ejecutar preprocesamiento si no existe
        if not clean_data.exists():
            print("⚙️  Ejecutando preprocesamiento...")
            subprocess.run(["python","src/preprocessing.py","data/raw/cumulative.csv","data/processed/exoplanets_clean.csv"])
        
        # Ejecutar feature engineering si no existe
        if not features_data.exists():
            print("⚙️  Generando características...")
            subprocess.run(["python","src/feature_engineering.py","data/processed/exoplanets_clean.csv","data/processed/exoplanets_features.csv"])
        
        # Entrenar modelo si no existe
        if not model_file.exists():
            print("⚙️  Entrenando modelo...")
            subprocess.run(["python","src/model.py","data/processed/exoplanets_features.csv","models/exoplanet_model.joblib"])
        
        print("🚀 Iniciando aplicación Streamlit...")
        subprocess.run(["streamlit","run","webapp/app.py"])
    else:
        print("""
Comandos disponibles:
  train_all       - Entrena los 3 modelos (KEPLER, TESS, K2)
  train_kepler    - Entrena solo modelo KEPLER
  train_tess      - Entrena solo modelo TESS
  train_k2        - Entrena solo modelo K2
  preprocess      - Preprocesa datos de KEPLER
  features        - Genera características
  train           - Entrena modelo (legacy)
  runstreamlit    - Inicia la aplicación web
        """)

if __name__ == "__main__":
    main()
