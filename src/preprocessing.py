"""
preprocessing.py
Lee el CSV original de la NASA (puede contener metadatos),
detecta dónde empieza la tabla, convierte a numérico, imputa y
genera un CSV limpio listo para feature engineering.
Uso:
    python src/preprocessing.py data/raw/cumulative.csv data/processed/exoplanets_clean.csv
"""
import sys, io, csv
import pandas as pd
import numpy as np
from pathlib import Path

def detect_table_start(path, hints=['rowid,', 'toi,', 'pl_name,']):
    """Detecta el inicio de la tabla en archivos de diferentes misiones"""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line_lower = line.strip().lower()
            for hint in hints:
                if line_lower.startswith(hint.lower()):
                    return i
    return None

def parse_table_from(path, start_idx):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    text = ''.join(lines[start_idx:])
    # usar csv.DictReader para preservar columnas textuales
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    df_raw = pd.DataFrame(rows)
    # además cargamos con pandas para manejo numérico robusto
    df_pd = pd.read_csv(io.StringIO(text), sep=',', engine='python', on_bad_lines='warn')
    # combinar: prioridad numérica de df_pd, textual de df_raw
    for col in df_pd.columns:
        if col in df_raw.columns:
            df_pd[col] = df_pd[col]
    return df_pd, df_raw

def to_numeric_columns(df):
    # Columnas que deben preservarse como texto (identificadores)
    text_columns = [
        'kepoi_name', 'kepler_name', 'kepid',  # KEPLER
        'toi', 'tid',  # TESS
        'pl_name', 'hostname',  # K2
        'koi_disposition', 'tfopwg_disp', 'disposition'  # Disposiciones
    ]
    
    for col in df.columns:
        # Saltar columnas de texto/identificadores
        if col in text_columns:
            continue
        try:
            # eliminar comas de miles y parsear
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')
        except Exception:
            pass
    return df

def impute_median(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    return df

def add_target_from_raw(df, df_raw):
    """Detecta automáticamente la misión y extrae las etiquetas correctas"""
    
    # KEPLER: usa koi_disposition
    if 'koi_disposition' in df_raw.columns:
        dispo = df_raw['koi_disposition'].fillna('').astype(str).str.strip().str.upper()
        mapping = {'CONFIRMED':'confirmed','CANDIDATE':'candidate','FALSE POSITIVE':'false_positive'}
        df['koi_disposition_raw'] = dispo.map(mapping).fillna('unknown').values
        df['target_label'] = df['koi_disposition_raw'].map({'confirmed':2,'candidate':1,'false_positive':0,'unknown':-1})
        print(f"✓ KEPLER detectado - Etiquetas procesadas")
    
    # TESS: usa tfopwg_disp (CP=Confirmed Planet, PC=Planet Candidate, FP=False Positive, KP=Known Planet)
    elif 'tfopwg_disp' in df_raw.columns:
        dispo = df_raw['tfopwg_disp'].fillna('').astype(str).str.strip().str.upper()
        mapping = {'CP':'confirmed', 'KP':'confirmed', 'PC':'candidate', 'FP':'false_positive'}
        df['koi_disposition_raw'] = dispo.map(mapping).fillna('unknown').values
        df['target_label'] = df['koi_disposition_raw'].map({'confirmed':2,'candidate':1,'false_positive':0,'unknown':-1})
        print(f"✓ TESS detectado - Etiquetas procesadas (CP/KP→Confirmed, PC→Candidate, FP→False Positive)")
    
    # K2: usa disposition
    elif 'disposition' in df_raw.columns:
        dispo = df_raw['disposition'].fillna('').astype(str).str.strip().str.upper()
        mapping = {'CONFIRMED':'confirmed','CANDIDATE':'candidate','FALSE POSITIVE':'false_positive'}
        df['koi_disposition_raw'] = dispo.map(mapping).fillna('unknown').values
        df['target_label'] = df['koi_disposition_raw'].map({'confirmed':2,'candidate':1,'false_positive':0,'unknown':-1})
        print(f"✓ K2 detectado - Etiquetas procesadas")
    
    else:
        df['koi_disposition_raw'] = 'unknown'
        df['target_label'] = -1
        print(f"⚠ No se encontró columna de disposición - Etiquetas desconocidas")
    
    return df

def apply_log_transform(df, cols):
    for c in cols:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
            minv = np.nanmin(arr)
            shift = -minv + 1e-6 if np.isfinite(minv) and minv <= 0 else 0.0
            df[c + '_log1p'] = np.log1p(arr + shift)
    return df

def main(in_path, out_path):
    print("Leyendo:", in_path)
    start = detect_table_start(in_path)
    if start is None:
        raise RuntimeError("No pude detectar inicio de tabla. Revisa el CSV.")
    df_pd, df_raw = parse_table_from(in_path, start)
    df_pd = to_numeric_columns(df_pd)
    df_pd = impute_median(df_pd)
    df_pd = add_target_from_raw(df_pd, df_raw)
    skew_cols = ['koi_period','koi_duration','koi_prad','koi_depth','koi_model_snr','koi_num_transits','koi_ror']
    df_pd = apply_log_transform(df_pd, skew_cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_pd.to_csv(out_path, index=False)
    print("Guardado preprocesado en:", out_path)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv)>1 else "data/raw/cumulative.csv"
    outp = sys.argv[2] if len(sys.argv)>2 else "data/processed/exoplanets_clean.csv"
    main(inp, outp)
