import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

def preprocesar_datos(
    df,
    eliminar_columnas_alta_nulos=True,
    umbral_nulos=0.65,
    imputacion=True,
    estrategia_imputacion='median',
    tratar_outliers=True,
    metodo_outliers='IQR',
    escalar=True,
    metodo_escalado='standard'
):
    df = df.copy()

    df['grade_code'] = pd.to_numeric(df['grade_code'], errors='coerce')
    df = df[df['grade_code'].notna()]
    df['target_bin'] = df['grade_code'].astype(int).apply(lambda x: 1 if x >= 7 else 0)

    if 'fecha_constitucion' in df.columns:
        df = df.drop(columns='fecha_constitucion')

    if 'codigo_empresa' in df.columns:
        df = df.drop(columns='codigo_empresa')

    if eliminar_columnas_alta_nulos:
        df = df.loc[:, df.isnull().mean() < umbral_nulos]

    if imputacion:
        imp = SimpleImputer(strategy=estrategia_imputacion)
        cols_target = ['grade_code', 'target_bin']
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col not in cols_target]
        df[numeric_cols] = imp.fit_transform(df[numeric_cols])

    if tratar_outliers:
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['grade_code', 'target_bin']]
        for col in numeric_cols:
            if metodo_outliers == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower, upper)
            elif metodo_outliers == 'zscore':
                z = (df[col] - df[col].mean()) / df[col].std()
                df[col] = np.where(z > 3, df[col].mean(), df[col])
                df[col] = np.where(z < -3, df[col].mean(), df[col])

    if escalar:
        scaler = StandardScaler() if metodo_escalado == 'standard' else MinMaxScaler()
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['grade_code', 'target_bin']]
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def seleccionar_variables(df, target_col, umbral_varianza=0.01, umbral_correlacion=0.95):
    df = df.copy()

    posibles_targets = ['grade_code', 'target_bin', 'grade_group']
    posibles_targets.remove(target_col)
    df = df.drop(columns=[col for col in posibles_targets if col in df.columns])

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in columnas_numericas:
        columnas_numericas.append(target_col)
    df = df[columnas_numericas]

    selector = VarianceThreshold(threshold=umbral_varianza)
    features = df.drop(columns=[target_col])
    selector.fit(features)
    columnas_baja_varianza = features.columns[~selector.get_support()]
    print(f"Eliminando por baja varianza: {list(columnas_baja_varianza)}")
    df = df.drop(columns=columnas_baja_varianza)

    corr_matrix = df.drop(columns=[target_col]).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    columnas_alta_correlacion = [col for col in upper.columns if any(upper[col] > umbral_correlacion)]
    print(f"Eliminando por alta correlacion: {columnas_alta_correlacion}")
    df = df.drop(columns=columnas_alta_correlacion)

    return df