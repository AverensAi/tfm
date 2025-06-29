import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
import os
from glob import glob
import re

def plot_confusion_matrix(y_true, y_pred, model_name, suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.savefig(f"outputs/conf_matrix_{model_name}{suffix}.png")
    plt.close()


from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_prob, model_name, suffix=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"Curva ROC - {model_name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(f"outputs/roc_curve_{model_name}{suffix}.png")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, suffix=""):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_idx])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in sorted_idx], 
                   rotation=90)
        plt.title(f"Importancia de Variables - {model_name}")
        plt.tight_layout()
        plt.savefig(f"outputs/feature_importance_{model_name}{suffix}.png")
        plt.close()


def generar_tabla_comparativa_f1(output_folder="outputs"):
    import pandas as pd
    import os
    from glob import glob
    import re

    csv_files = glob(os.path.join(output_folder, "metricas_*.csv"))

    df_list = []
    for file in csv_files:
        dataset_name = re.search(r"metricas_(.*)\.csv", file).group(1)
        df = pd.read_csv(file)
        df["dataset_name"] = dataset_name
        df_list.append(df)

    if not df_list:
        print("⚠️ No se encontraron archivos de métricas.")
        return None

    df_all = pd.concat(df_list, ignore_index=True)

    tabla = df_all.pivot_table(
        index="modelo",
        columns="dataset_name",
        values="f1_score",
        aggfunc="max"
    )

    # Guardar redondeada
    tabla_path = os.path.join(output_folder, "tabla_comparativa_f1.csv")
    tabla.round(2).to_csv(tabla_path)
    print(f"✅ Tabla comparativa guardada en: {tabla_path}")

    # Mostrar el mejor modelo por dataset
    mejores_por_dataset = tabla.idxmax()
    print("\n🏆 Mejor modelo por dataset:")
    for dataset, modelo in mejores_por_dataset.items():
        f1 = tabla.loc[modelo, dataset]
        print(f"- {dataset}: {modelo} (F1 = {f1:.2f})")

    # Mejor modelo global
    mejor_global = tabla.stack().idxmax()
    mejor_modelo_global, mejor_dataset_global = mejor_global
    f1_global = tabla.loc[mejor_modelo_global, mejor_dataset_global]
    print(f"\n🌟 Mejor modelo global: {mejor_modelo_global} con {mejor_dataset_global} (F1 = {f1_global:.2f})")

    return tabla
