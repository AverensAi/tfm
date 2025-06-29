import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os
import joblib
import random


random.seed(42)
np.random.seed(42)

# Crear carpeta para guardar gráficos y métricas
os.makedirs("outputs", exist_ok=True)

# Funciones personalizadas
from config import cargar_parametros_modelos
from preprocesamiento import preprocesar_datos, seleccionar_variables
from generadores import generar_datos_ctgan
from plots import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, generar_tabla_comparativa_f1

def entrenar_modelos(X, y, dataset_name='Original', scoring='f1', test_size=0.2, random_state=42):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    parametros_modelos = cargar_parametros_modelos()

    modelos = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),  # Este no usa random_state
    'SVC': SVC(probability=True, random_state=42),
    'NaiveBayes': GaussianNB(),     # No tiene random_state
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
    }

    resultados = []

    for nombre, modelo in modelos.items():
        with mlflow.start_run(run_name=f"{dataset_name} - {nombre}", nested=True):
            print(f"\n Entrenando {nombre} con {dataset_name}")
            try:
                params = parametros_modelos.get(nombre, {})
                grid = GridSearchCV(modelo, params, cv=3, scoring=scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Guardar métricas y run_id
                resultados.append({
                    "dataset": dataset_name,
                    "modelo": nombre,
                    "accuracy": acc,
                    "f1_score": f1,
                    "precision": prec,
                    "recall": recall,
                    "run_id": mlflow.active_run().info.run_id
                })

                # Probabilidades para ROC
                if hasattr(best_model, "predict_proba"):
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                elif hasattr(best_model, "decision_function"):
                    y_proba = best_model.decision_function(X_test)
                else:
                    y_proba = None

                # Gráficos
                suffix = f"_{dataset_name}_{nombre}"
                plot_confusion_matrix(y_test, y_pred, nombre, suffix=suffix)
                if y_proba is not None:
                    plot_roc_curve(y_test, y_proba, nombre, suffix=suffix)
                if hasattr(best_model, "feature_importances_"):
                    plot_feature_importance(best_model, X.columns, nombre, suffix=suffix)

                # Guardar artefactos
                mlflow.log_artifact(f"outputs/conf_matrix_{nombre}{suffix}.png")
                if y_proba is not None:
                    mlflow.log_artifact(f"outputs/roc_curve_{nombre}{suffix}.png")
                if hasattr(best_model, "feature_importances_"):
                    mlflow.log_artifact(f"outputs/feature_importance_{nombre}{suffix}.png")

                # Log de parámetros y métricas
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_param("model", nombre)
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", recall)

                # Guardar modelo
                input_example = X_test.iloc[[0]]
                signature = mlflow.models.infer_signature(X_test, y_pred)
                mlflow.sklearn.log_model(best_model, "model", input_example=input_example, signature=signature)

                # Registrar modelo
                try:
                    mlflow.register_model(
                        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                        name=f"{nombre}_{dataset_name}_{mlflow.active_run().info.run_id[:6]}"
                    )
                    joblib.dump(best_model, f"outputs/{nombre}_{dataset_name}.pkl")
                except Exception as reg_err:
                    print(f" Registro no disponible para {nombre}: {reg_err}")

                print(classification_report(y_test, y_pred))

            except Exception as e:
                print(f" Error con {nombre}: {repr(e)}")

    # Guardar resultados en CSV
    df_resultados = pd.DataFrame(resultados)
    csv_path = os.path.join("outputs", f"metricas_{dataset_name}.csv")
    df_resultados.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    # Buscar mejor modelo para registrar como global
    mejor_resultado = max(resultados, key=lambda x: x["f1_score"])
    param_name = f"mejor_modelo_global_{dataset_name.replace('+', '_')}"
    mlflow.log_param(param_name, f"{mejor_resultado['modelo']}_{mejor_resultado['dataset']}")
    print(f"\n Mejor modelo global: {mejor_resultado['modelo']} entrenado con {mejor_resultado['dataset']} (F1={mejor_resultado['f1_score']:.2f})")

    try:
        mlflow.register_model(
            model_uri=f"runs:/{mejor_resultado['run_id']}/model",
            name="MejorModeloGlobal"
        )
        
    except Exception as e:
        print(f" No se pudo registrar el mejor modelo global: {e}")

if __name__ == "__main__":
    mlflow.set_experiment("TFM_CreditScoring")
    with mlflow.start_run(run_name="TFM completo"):
        df = pd.read_parquet("companies_T_anon.parquet")

        df = preprocesar_datos(df)
        df = seleccionar_variables(df, target_col="target_bin")
        X = df.drop(columns=["target_bin"])
        y = df["target_bin"]

        print(" Datos originales")
        entrenar_modelos(X, y, dataset_name="Original")

        print(" Datos con SMOTE")
        X_smote, y_smote = SMOTE(random_state=42).fit_resample(X, y)
        entrenar_modelos(X_smote, y_smote, dataset_name="SMOTE")

        print(" Datos con CTGAN")
        X_ctgan, y_ctgan = generar_datos_ctgan(X, y, selected_features=X.columns.tolist())
        entrenar_modelos(X_ctgan, y_ctgan, dataset_name="CTGAN")

        print(" Datos con CTGAN + SMOTE")
        X_ctgan_smote, y_ctgan_smote = SMOTE(random_state=42).fit_resample(X_ctgan, y_ctgan)
        entrenar_modelos(X_ctgan_smote, y_ctgan_smote, dataset_name="CTGAN+SMOTE")

        print(" Datos con SMOTE + CTGAN")
        X_smote_ctgan, y_smote_ctgan = generar_datos_ctgan(X_smote, y_smote, selected_features=X.columns.tolist())
        entrenar_modelos(X_smote_ctgan, y_smote_ctgan, dataset_name="SMOTE+CTGAN")

        print("\n Generando tabla comparativa final...")
        generar_tabla_comparativa_f1()
