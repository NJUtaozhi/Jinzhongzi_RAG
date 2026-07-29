FROM python:3.10-slim

WORKDIR /app

COPY docker/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ /app/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
