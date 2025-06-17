#!/bin/bash

echo "ローカルでアプリケーションをテスト中..."

# Dockerイメージをビルド
echo "Dockerイメージをビルド中..."
docker build -t concrete-crack-detection .

# コンテナを実行
echo "コンテナを起動中..."
docker run -p 8080:8080 concrete-crack-detection

# ブラウザで http://localhost:8080 にアクセスしてください