#Dockerfile
# 使用官方 Python 基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制本地的 requirements.txt 文件到 Docker 容器中
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt

# 复制保存的模型到容器中
COPY ./saved_model ./saved_model

# 将 bert-base-chinese 模型文件复制到容器中
COPY ./bert-models /app/bert-models


ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COMPILE=1
ENV PIP_CONCURRENCY=1




# 复制本地代码到容器中
COPY . .

EXPOSE 5000
# 开放 Flask 应用使用的端口

CMD ["python", "app.py"]
# 使用 Flask 应用作为容器的主要入口点




