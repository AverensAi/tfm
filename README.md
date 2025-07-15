### TFM - Credit Scoring con Small Data y MLOps

Este proyecto implementa un pipeline de clasificación para credit scoring utilizando técnicas de generación sintética (SMOTE, CTGAN) y modelos clásicos de machine learning. Incluye integración con MLflow para el seguimiento de experimentos y Docker para entornos reproducibles.

El repositorio completo está disponible en [GitHub](https://github.com/AverensAi/tfm).

## Estructura del proyecto

- train.py: Script principal de entrenamiento y evaluación.
- plots.py: Funciones para generar gráficos de matriz de confusión, curvas ROC e importancia de características.
- generadores.py: Generación de datos sintéticos con CTGAN.
- preprocesamiento.py: Limpieza, selección y transformación de datos.
- Dockerfile y docker-compose.yml: Configuración de contenedor y entorno MLflow.
- requirements.txt: Librerías necesarias.
- outputs/: Visualizaciones y métricas exportadas en CSV.
- mlruns/: Directorio interno de MLflow.
- companies_T_anon.parquet: Dataset de entrada anonimizado.
- README.md: Este archivo.

## Cómo ejecutar

### 1. Clonar el repositorio

```bash
git clone https://github.com/AverensAi/tfm.git
cd tfm
```

### 2. Construir la imagen Docker

```bash
docker build -t tfm .
```

### 3. Ejecutar el entrenamiento

```bash
docker run --rm -v $(pwd):/app -w /app tfm python train.py
```

> Asegúrate de tener instalado Docker: https://www.docker.com/products/docker-desktop

### 4. Interpretar el modelo con SHAP

```bash
docker run --rm -v $(pwd):/app -w /app tfm python interpretacion_shap.py
```

Esto generará:

- outputs/shap_summary_mlp.png: Gráfico SHAP summary.
- outputs/shap_bar_mlp.png: Gráfico de barras de importancia.
- outputs/dep_<feature>.png: Gráficos de dependencia por variable.
- outputs/dependence_grid.png: Subplots combinados de dependencia.
- outputs/force_instance0.png: Force plot de una instancia.
- outputs/decision_plot.png: Decision plot con top‑features.

### 5. Visualizar resultados con MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Luego abre tu navegador en http://localhost:5000

Si no funciona con el port 5000, pruebe con otro (por ejemplo, 5001, 5002...)