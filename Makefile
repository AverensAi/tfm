# Ejecuta solo SHAP MLP
shap:
	docker-compose up shap_mlp

# Ejecuta solo MLflow UI
mlflow:
	docker-compose up mlflow

# Ejecuta ambos servicios
all:
	docker-compose up

# Limpia contenedores detenidos y volumenes
clean:
	docker-compose down --volumes --remove-orphans
