FROM python:3.10-slim

# OpenFace 依赖（基础系统库）
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/vision/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制服务代码
COPY vision_service/ /app/

# 注意：OpenFace 工具需要单独挂载或安装在容器中
# 简化方案：OpenFace 在宿主机运行，用卷挂载方式调用

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
