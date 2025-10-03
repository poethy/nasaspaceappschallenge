"""
Funciones utilitarias: gráficas simples y métricas.
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(df, col, bins=50):
    plt.figure(figsize=(6,4))
    df[col].hist(bins=bins)
    plt.title(col)
    plt.tight_layout()
    return plt
