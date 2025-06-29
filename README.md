# TFM - Credit Scoring con Small Data y MLOps

Este proyecto implementa un pipeline de clasificación para credit scoring utilizando técnicas de generación sintética (SMOTE, CTGAN) y modelos clásicos de machine learning. Incluye integración con MLflow para el seguimiento de experimentos y Docker para entornos reproducibles.

## Estructura del proyecto

- `train.py`: Script principal de entrenamiento y evaluación.
- `plots.py`: Funciones para generar gráficos de matriz de confusión, curvas ROC e importancia de características.
- `generadores.py`: Generación de datos sintéticos con CTGAN.
- `preprocesamiento.py`: Limpieza, selección y transformación de datos.
- `Dockerfile` y `docker-compose.yml`: Configuración de contenedor y entorno MLflow.
- `requirements.txt`: Librerías necesarias.
- `outputs/`: Visualizaciones y métricas exportadas en CSV.
- `mlruns/`: Directorio interno de MLflow.
- `companies_T_anon.parquet`: Dataset de entrada.
- `README.md`: Este archivo.

## Cómo ejecutar

```bash
# Construir imagen
docker build -t tfm_project .

# Ejecutar entrenamiento dentro del contenedor
docker run --rm -v $(pwd):/app -w /app tfm_project python train.py
