from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
from dotenv import load_dotenv
import os

# 1. יצירת האפליקציה בראש הקובץ
app = FastAPI()
load_dotenv() 

# 2. הוספת ה-CORS מיד בהתחלה - זה התיקון העיקרי ל"אדום" בדפדפן
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # מאשר לכל הדומיינים (כולל GitHub Pages)
    allow_methods=["*"],
    allow_headers=["*"], # מאשר שליחת headers כמו x-api-key
)

# 3. הגדרות (עדיפות למפתח קבוע אם אין לך כוח להסתבך עם .env ב-Render)
API_KEY = 'SaharY0011'
API_URL = 'https://imageprocess-9zk1.onrender.com/predict'

# -------------------------------
# Model Architecture & Loading
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Dropout(0.6),
    nn.Linear(128*14*14, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 2)
).to(device)

# טעינה בטוחה כדי למנוע קריסה של השרת
try:
    model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load model weights: {e}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Endpoints
# -------------------------------

class ImageRequest(BaseModel):
    image_base64: str

@app.get("/config")
def get_config():
    return {"api_url": API_URL}

@app.post("/predict")
async def predict(request: ImageRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        x = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            y = model(x)
            prob = torch.softmax(y, dim=1)
            label_idx = int(torch.argmax(prob))
            confidence = float(torch.max(prob))

        class_names = ["AI-generated", "Real"]
        label_name = class_names[label_idx]

        return {
            "label": label_idx,
            "label_name": label_name,
            "confidence": round(confidence, 2),
            "certainty": "High" if confidence >= 0.7 else "Medium" if confidence >= 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
