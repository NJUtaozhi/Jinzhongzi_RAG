FROM python:3.10-slim

WORKDIR /app

COPY docker/agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_service/ /app/

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
