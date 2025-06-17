# コンクリートひび割れ検出システム

AIを活用してコンクリート表面のひび割れを自動検出するWebアプリケーションです。ResNet50ベースの深層学習モデルを使用し、3種類のコンクリート表面（デッキ、舗装、壁）に対応しています。

## 🎯 主な機能

- **3種類の表面タイプに対応**
  - コンクリートデッキ (D)
  - 舗装 (P)
  - 壁 (W)
- **リアルタイム画像分析**
- **確信度の可視化表示**
- **高精度な検出** (90%以上の精度)

## 🛠 技術スタック

- **バックエンド**: FastAPI
- **深層学習**: PyTorch, ResNet50
- **フロントエンド**: HTML, CSS, JavaScript
- **デプロイ**: Google Cloud Run

## 📊 モデル性能

各表面タイプのモデル精度：
- デッキ (D): 90.90%
- 舗装 (P): 93.84%
- 壁 (W): 90.24%

## 🚀 使い方

1. Webアプリケーションにアクセス
2. 検査する表面タイプを選択（デッキ/舗装/壁）
3. コンクリート表面の画像をアップロード
4. 「分析開始」ボタンをクリック
5. ひび割れの有無と確信度が表示されます

## 📁 プロジェクト構成

```
.
├── app.py                 # FastAPIメインアプリケーション
├── models/               # 学習済みモデル
│   ├── best_model_D.pth  # デッキ用モデル
│   ├── best_model_P.pth  # 舗装用モデル
│   └── best_model_W.pth  # 壁用モデル
├── results/              # 学習結果
│   ├── classification_report_*.txt
│   ├── confusion_matrix_*.png
│   └── training_history_*.png
├── templates/
│   └── index.html        # フロントエンドUI
├── train.ipynb           # モデル学習コード
└── README.md
```

## 🔧 ローカル環境での実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# サーバーの起動
python app.py
```

アプリケーションは `http://localhost:8080` で起動します。

## ☁️ Google Cloud Runへのデプロイ

### 前提条件
1. Google Cloud プロジェクトの作成
2. GitHub リポジトリへのアクセス

### セットアップ手順

1. **Google Cloud でサービスアカウントを作成**
   ```bash
   # サービスアカウントを作成
   gcloud iam service-accounts create github-actions --display-name "GitHub Actions"
   
   # 必要な権限を付与
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/run.admin"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   
   # サービスアカウントキーを作成
   gcloud iam service-accounts keys create key.json \
     --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

2. **GitHub Secretsを設定**
   - `GCP_PROJECT_ID`: Google CloudプロジェクトID
   - `GCP_SA_KEY`: サービスアカウントキー（key.jsonの内容）

3. **mainブランチにpushすると自動デプロイ**

## 📝 学習データ

SDNET2018データセットを使用してモデルを学習しました。各表面タイプについて、ひび割れあり/なしの画像を使用し、データ拡張を適用して汎化性能を向上させています。

## 🤝 貢献

プルリクエストやイシューの作成を歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。