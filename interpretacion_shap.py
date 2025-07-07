#!/usr/bin/env python3
"""
interpretacion_shap.py — Interpretación de un modelo MLP entrenado con datos CTGAN + SMOTE

Ejecuta desde terminal o VS Code:
    python interpretacion_shap.py

Genera:
    models/mlp_ctgan_smote.pkl        # modelo entrenado
    outputs/shap_summary_mlp.png      # summary plot SHAP
    outputs/dep_<feature>.png         # dependence plots de top‑features
    outputs/force_instance0.png       # force plot de la fila 0

Ajusta rutas o parámetros según el proyecto.
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------
import math
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

print(f"SHAP version: {shap.__version__}")

# Módulos propios
from preprocesamiento import preprocesar_datos, seleccionar_variables
from generadores import generar_datos_ctgan

# --------------------------------------------------
# Parámetros
# --------------------------------------------------
DATA_PATH = "companies_T_anon.parquet"
RANDOM_STATE = 1
N_SYNTH_SAMPLES = 300          # n° de filas sintéticas CTGAN
N_SHAP_BACKGROUND = 100        # n° filas fondo PermutationExplainer
N_SHAP_DISPLAY = 10            # features en summary plot

# --------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------

def set_seeds(seed: int = 1):
    """Fija las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------
# Pipeline principal
# --------------------------------------------------

def main() -> None:
    set_seeds(RANDOM_STATE)

    # 1) Cargar datos -------------------------------------------------------
    df = pd.read_parquet(DATA_PATH)

    # 2) Preprocesamiento ---------------------------------------------------
    df = preprocesar_datos(df)
    df = seleccionar_variables(df, target_col="target_bin")

    y = df["target_bin"].reset_index(drop=True)
    X = df.drop(columns=["target_bin"]).reset_index(drop=True)

    # 3) Generar datos sintéticos con CTGAN ---------------------------------
    print("Generando datos con CTGAN…")
    X_ctgan_scaled, y_ctgan = generar_datos_ctgan(
        X,
        y,
        target_col="target_bin",
        selected_features=X.columns,
        n_samples=N_SYNTH_SAMPLES,
        epochs=300,
    )

    # 4) Balancear con SMOTE -------------------------------------------------
    print("Aplicando SMOTE…")
    X_bal, y_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_ctgan_scaled, y_ctgan
    )

    # 5) Train/test split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=0.2,
        stratify=y_bal,
        random_state=RANDOM_STATE,
    )

    feature_names = list(X.columns)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # 6) Entrenar MLP -------------------------------------------------------
    print("Entrenando MLP…")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train_df, y_train)

    # 7) Guardar modelo -----------------------------------------------------
    Path("models").mkdir(exist_ok=True)
    joblib.dump(mlp, "models/mlp_ctgan_smote.pkl")

    # 8) SHAP: PermutationExplainer ----------------------------------------
    print("Calculando valores SHAP… (puede tardar)")
    background = resample(
        X_train_df, n_samples=N_SHAP_BACKGROUND, random_state=RANDOM_STATE
    )
    explainer = shap.explainers.Permutation(mlp.predict_proba, background)

    exp = explainer(X_test_df)  # objeto Explanation
    shap_values = exp.values  # (n_samples, n_features, n_classes)

    print("shap_values shape:", shap_values.shape)
    print("X_test_df shape:", X_test_df.shape)

    Path("outputs").mkdir(exist_ok=True)

    # 9) Summary plot -------------------------------------------------------
    shap.summary_plot(
        shap_values[:, :, 1],  # clase positiva
        X_test_df,
        max_display=N_SHAP_DISPLAY,
        show=False,
        plot_size=(10, 6),
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_mlp.png", dpi=300)
    plt.close()

    # 10) Dependence plots de top‑features ---------------------------------
    shap_matrix = shap_values[:, :, 1]  # Solo para la clase positiva
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-N_SHAP_DISPLAY:][::-1]

    for idx in top_idx:
        feat_name = feature_names[int(idx)]
        shap.dependence_plot(
            feat_name,
            shap_matrix,
            X_test_df,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(f"outputs/dep_{feat_name}.png", dpi=300)
        plt.close()

    # 11) Force plot de la primera fila ------------------------------------
    shap.initjs()
    # base value para la clase positiva (índice 1)
    base_val_pos = exp.base_values[0][1] if exp.base_values.ndim == 2 else exp.base_values[1]

    shap.force_plot(
        base_value=base_val_pos,
        shap_values=shap_matrix[0],
        features=X_test_df.iloc[0],
        show=False,
        matplotlib=True,
    )
    plt.savefig("outputs/force_instance0.png", dpi=300)
    plt.close()

    # 12) Bar plot de importancia -----------------------------------------
    shap.summary_plot(
        shap_matrix,
        X_test_df,
        plot_type="bar",
        max_display=N_SHAP_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_bar_mlp.png", dpi=300)
    plt.close()

    # 13) Subplots de dependencia (top‑N) ----------------------------------
    n_features = N_SHAP_DISPLAY
    top_feature_names = [feature_names[i] for i in top_idx]

    cols = 3
    rows = math.ceil(n_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(top_feature_names):
        shap.dependence_plot(
            feature,
            shap_matrix,
            X_test_df,
            ax=axes[i],
            show=False,
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("outputs/dependence_grid.png", dpi=300)
    plt.close()

    # 14) Decision plot ----------------------------------------------------
    shap.decision_plot(
        base_val_pos,
        shap_matrix,
        X_test_df,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig("outputs/decision_plot.png", dpi=300)
    plt.close()

    print("Listo ✅  — Gráficas guardadas en 'outputs/'")


if __name__ == "__main__":
    main()
