FROM python:3.10-slim-bookworm

WORKDIR /app

COPY . .

# Update packages and clean up cache to reduce image size and fix CVEs
RUN apt-get update && apt-get upgrade -y && apt-get install -y build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["python", "train.py"]
