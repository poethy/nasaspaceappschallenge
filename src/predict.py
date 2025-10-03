"""
predict.py
Carga un modelo guardado y predice sobre un CSV de entrada.
Uso:
    python src/predict.py models/exoplanet_model.joblib data/new_to_predict.csv out_preds.csv
"""
import sys, joblib
import pandas as pd, numpy as np
from pathlib import Path

def predict(model_path, in_csv, out_csv):
    model = joblib.load(model_path)
    scaler = joblib.load(str(Path(model_path).with_suffix('.scaler.joblib')))
    features = joblib.load(str(Path(model_path).with_suffix('.features.joblib')))
    df = pd.read_csv(in_csv)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].copy()
    # align features
    X_model = pd.DataFrame(0, index=X.index, columns=features)
    for c in X.columns:
        if c in X_model.columns:
            X_model[c] = X[c]
    Xs = scaler.transform(X_model.values)
    probs = model.predict_proba(Xs) if hasattr(model, 'predict_proba') else model.predict(Xs)
    if probs.ndim == 2 and probs.shape[1] == 3:
        out = pd.DataFrame(probs, columns=['prob_falsepos','prob_candidate','prob_confirmed'])
    else:
        out = pd.DataFrame({'score': probs})
    out = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    out.to_csv(out_csv, index=False)
    print("Predicciones guardadas en:", out_csv)

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv)>1 else "models/exoplanet_model.joblib"
    in_csv = sys.argv[2] if len(sys.argv)>2 else "data/processed/exoplanets_features.csv"
    out_csv = sys.argv[3] if len(sys.argv)>3 else "preds.csv"
    predict(model_path, in_csv, out_csv)
