# Python 3.9のスリムイメージを使用
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコピー
COPY . .

# Cloud Runはポート8080を使用
ENV PORT=8080

# アプリケーションを起動
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT