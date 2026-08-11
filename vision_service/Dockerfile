FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 不安装 OpenFace，验证服务可正常启动
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
