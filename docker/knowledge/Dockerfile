FROM python:3.10-slim

# curl 用于 Docker 健康检查
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/knowledge/requirements.txt .
# 升级 pip 然后装 CPU 版 torch（避免下载 600MB+ CUDA 包）；使用腾讯云 PyPI 镜像加速
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 torch --extra-index-url https://download.pytorch.org/whl/cpu -i https://mirrors.cloud.tencent.com/pypi/simple
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

COPY knowledge_service/ /app/

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
