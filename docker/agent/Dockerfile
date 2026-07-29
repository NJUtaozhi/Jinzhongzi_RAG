FROM python:3.10-slim

WORKDIR /app

# 使用成员3的完整依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 Agent 服务代码（成员3：根目录结构）
COPY main.py /app/
COPY factories/ /app/factories/
COPY orchestration/ /app/orchestration/
COPY services/ /app/services/
COPY tools/ /app/tools/

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
