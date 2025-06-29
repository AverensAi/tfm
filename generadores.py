from ctgan import CTGAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import torch
import random
import numpy as np

def fijar_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generar_datos_ctgan(X, y, target_col="target_bin", selected_features=None, n_samples=300, epochs=300):
    fijar_seed(42)
    df_ctgan = pd.DataFrame(X, columns=selected_features)

    # 1. Eliminar columnas datetime
    datetime_cols = df_ctgan.select_dtypes(include="datetime64").columns
    if len(datetime_cols) > 0:
        print(f"Eliminando columnas datetime: {list(datetime_cols)}")
        df_ctgan.drop(columns=datetime_cols, inplace=True)

    # 2. Imputar nulos (para que CTGAN no falle)
    imputer = SimpleImputer(strategy="median")
    df_ctgan[df_ctgan.columns] = imputer.fit_transform(df_ctgan)

    # 3. Añadir target
    df_ctgan[target_col] = y.reset_index(drop=True)

    # 4. Detectar columnas categóricas (object)
    discrete_cols = df_ctgan.select_dtypes(include='object').columns.tolist()
    if target_col not in discrete_cols:
        discrete_cols.append(target_col)

    # 5. Entrenar CTGAN
    ctgan = CTGAN(epochs=epochs)
    ctgan.fit(df_ctgan, discrete_columns=discrete_cols)

    # 6. Generar datos sintéticos
    synthetic_data = ctgan.sample(n_samples)
    synthetic_data[target_col] = synthetic_data[target_col].round()
    synthetic_data[target_col] = synthetic_data[target_col].fillna(0).astype(int)

    print("Distribución de clase sintética:")
    print(synthetic_data[target_col].value_counts())

    # 7. Concatenar reales y sintéticos
    df_combined = pd.concat([df_ctgan, synthetic_data], ignore_index=True)
    X_combined = df_combined.drop(columns=[target_col])
    y_combined = df_combined[target_col]

    # 8. One-hot encoding y escalado
    X_combined = pd.get_dummies(X_combined, drop_first=True)
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)

    # Asegurar que y_combined no tiene nulos
    y_combined = y_combined.fillna(0).astype(int)

    return X_combined_scaled, y_combined
