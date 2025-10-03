"""
feature_engineering.py
Extrae features adicionales. Aquí hay dos rutas:
- Si tienes curvas de luz (series temporales): usar tsfresh y BLS.
- Si trabajas con catálogo (como el que nos diste): añadir ratio/relaciones y conservar columnas relevantes.

Uso:
    python src/feature_engineering.py data/processed/exoplanets_clean.csv data/processed/exoplanets_features.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Intentamos usar tsfresh si está instalado (opcional)
try:
    from tsfresh import extract_features
    have_tsfresh = True
except Exception:
    have_tsfresh = False

def basic_feature_engineering(df):
    # Añadir ratios y features simples derivados
    df = df.copy()
    if 'koi_depth' in df.columns and 'koi_ror' in df.columns:
        df['depth_over_ror'] = pd.to_numeric(df['koi_depth'], errors='coerce') / (pd.to_numeric(df['koi_ror'], errors='coerce')+1e-9)
    # ejemplo de relación periodo/duración
    if 'koi_period' in df.columns and 'koi_duration' in df.columns:
        df['period_over_duration'] = pd.to_numeric(df['koi_period'], errors='coerce') / (pd.to_numeric(df['koi_duration'], errors='coerce')+1e-9)
    # Fill NaNs generados
    df = df.fillna(0)
    return df

def main(in_csv, out_csv):
    print("Cargando:", in_csv)
    df = pd.read_csv(in_csv)
    df_feat = basic_feature_engineering(df)
    # Si tenemos series crudas o tsfresh: podríamos extraer features adicionales (no aplicado aquí por default)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_csv, index=False)
    print("Guardado features en:", out_csv)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv)>1 else "data/processed/exoplanets_clean.csv"
    outp = sys.argv[2] if len(sys.argv)>2 else "data/processed/exoplanets_features.csv"
    main(inp, outp)
