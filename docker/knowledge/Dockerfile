FROM python:3.10-slim

WORKDIR /app

COPY docker/knowledge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY knowledge_service/ /app/

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
