# Imagen base con Python
FROM python:3.9

# Crear directorio de trabajo
WORKDIR /app

# Copiar todos los archivos del proyecto
COPY . /app

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto para el MLflow UI
EXPOSE 5000

# Comando por defecto: ejecutar train.py
CMD ["python", "train.py"]
