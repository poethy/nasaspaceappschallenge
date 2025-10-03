import streamlit as st
import pandas as pd
st.title("Subir datos")
uploaded = st.file_uploader("Sube CSV con catálogo NASA", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write(df.head())
    if st.button("Guardar en data/raw/cumulative.csv"):
        df.to_csv("data/raw/cumulative.csv", index=False)
        st.success("Guardado en data/raw/cumulative.csv")
