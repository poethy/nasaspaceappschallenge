"""
model.py
Entrena un modelo básico (LightGBM) multiclass y guarda el modelo + lista de features + scaler.
Uso:
    python src/model.py data/processed/exoplanets_features.csv models/exoplanet_model.joblib
"""
import sys, joblib
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

def train(in_csv, out_model):
    print("Cargando:", in_csv)
    df = pd.read_csv(in_csv)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target_label' not in numeric_cols:
        raise RuntimeError("No se encontró 'target_label' en el dataset.")
    X = df[numeric_cols].copy()
    y = X.pop('target_label')
    mask = y >= 0
    X = X[mask]
    y = y[mask].astype(int)
    # eliminar columnas no útiles
    for c in ['rowid','kepid','tid','toi']:
        if c in X.columns:
            X.drop(columns=[c], inplace=True)
    features = X.columns.tolist()
    # escalado
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    
    # Decidir si usar GroupKFold o StratifiedKFold
    groups = df.loc[mask, 'kepid'] if 'kepid' in df.columns else None
    
    if groups is not None:
        print("✓ Usando GroupKFold (agrupando por kepid)")
        cv = GroupKFold(n_splits=3)
        split_args = (Xs, y, groups)
    else:
        print("✓ Usando StratifiedKFold (sin grupos)")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        split_args = (Xs, y)
    
    fold = 0
    models = []
    for tr, val in cv.split(*split_args):
        fold += 1
        Xtr, Xval = Xs[tr], Xs[val]
        ytr, yval = y.iloc[tr], y.iloc[val]
        model = LGBMClassifier(objective='multiclass', num_class=3, n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)
        model.fit(Xtr, ytr)
        preds = model.predict(Xval)
        print(f"Fold {fold} report:\\n", classification_report(yval, preds))
        models.append(model)
    # guardar último modelo, scaler y features
    joblib.dump(models[-1], out_model)
    joblib.dump(scaler, str(Path(out_model).with_suffix('.scaler.joblib')))
    joblib.dump(features, str(Path(out_model).with_suffix('.features.joblib')))
    print("Modelo guardado en:", out_model)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv)>1 else "data/processed/exoplanets_features.csv"
    out = sys.argv[2] if len(sys.argv)>2 else "models/exoplanet_model.joblib"
    train(inp, out)
