from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = FastAPI()

# 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# 表面タイプの設定
SURFACE_TYPES = {
    'D': 'コンクリートデッキ',
    'P': '舗装',
    'W': '壁'
}

# モデルの定義
def create_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    return model

# 各表面タイプのモデルを読み込む
print("Loading models...")
models_dict = {}
for surface_type in SURFACE_TYPES.keys():
    try:
        model = create_model()
        model_path = os.path.join('models', f'best_model_{surface_type}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models_dict[surface_type] = model
        print(f"Loaded model for {SURFACE_TYPES[surface_type]}")
    except Exception as e:
        print(f"Error loading model for {surface_type}: {str(e)}")

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "surface_types": SURFACE_TYPES.items()
        }
    )

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), surface_type: str = Form(...)):
    try:
        # 入力検証
        if surface_type not in SURFACE_TYPES:
            return {"error": "Invalid surface type"}
            
        if surface_type not in models_dict:
            return {"error": f"Model for {SURFACE_TYPES[surface_type]} is not loaded"}

        # 画像の読み込みと前処理
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # デバイスに転送
        image_tensor = image_tensor.to(device)
        
        # モデルの取得と予測
        model = models_dict[surface_type]
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            
            # 確率をCPUに移動してからnumpy変換
            prob_no_crack = float(probabilities[0].cpu().item()) * 100
            prob_crack = float(probabilities[1].cpu().item()) * 100
            
            result = {
                "surface_type": SURFACE_TYPES[surface_type],
                "predicted_class": "Crack" if predicted_class_idx == 1 else "No Crack",
                "probabilities": {
                    "No Crack": f"{prob_no_crack:.2f}",
                    "Crack": f"{prob_crack:.2f}"
                }
            }
        
        return result
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)