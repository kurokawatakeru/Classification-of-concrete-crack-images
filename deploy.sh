#!/bin/bash

# プロジェクトIDを設定（あなたのプロジェクトIDに変更してください）
PROJECT_ID="reac-463202"
REGION="asia-northeast1"
SERVICE_NAME="concrete-crack-detection"

echo "Google Cloud Runへのデプロイを開始します..."

# 1. Google Cloud CLIの認証確認
echo "認証状態を確認中..."
gcloud auth list

# 2. プロジェクトの設定
echo "プロジェクトを設定中..."
gcloud config set project $PROJECT_ID

# 3. Cloud Build APIとCloud Run APIを有効化
echo "必要なAPIを有効化中..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 4. Cloud Buildを使用してデプロイ
echo "Cloud Buildを実行中..."
gcloud builds submit --config cloudbuild.yaml

# 5. デプロイされたサービスのURLを取得
echo "サービスのURLを取得中..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)")
echo "デプロイ完了！"
echo "サービスURL: $SERVICE_URL"