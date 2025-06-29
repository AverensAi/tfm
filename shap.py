# app.py
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Cargar modelo y datos
model = joblib.load("SVC_CTGAN.pkl")
X_test = pd.read_parquet("X_test.parquet")

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Mostrar gráfico SHAP
st.title("Explicabilidad con SHAP")
st.write("Gráfico SHAP summary (clase 1):")
fig = plt.figure()
shap.summary_plot(shap_values[1], X_test, show=False)
st.pyplot(fig)
