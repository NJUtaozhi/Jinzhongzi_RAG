FROM python:3.10-slim

# curl 用于 Docker 健康检查
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ /app/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
