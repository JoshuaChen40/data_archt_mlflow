# 使用官方 Python 映像作為基礎
FROM python:3.8-slim

# 設定工作目錄
WORKDIR /mlflow

# 複製本地代碼到容器中
COPY . /mlflow

# 安裝 dbt
RUN pip install --no-cache-dir -r /mlflow/requirements.txt
