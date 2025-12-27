from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms

# -------------------------------
# FastAPI + CORS
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://saharyaccov.github.io"],  # רק הדומיין שלך
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------
# בקשה JSON
# -------------------------------
class ImageRequest(BaseModel):
    image_base64: str

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model Architecture
# -------------------------------
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

model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()

# -------------------------------
# Transform כמו בלידציה
# -------------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Endpoint לחיזוי עם API Key
# -------------------------------
API_KEY = "SaharY0011"

@app.post("/predict")
async def predict(request: ImageRequest, x_api_key: str = Header(None)):
    # בדיקה אם ה-API Key נכון
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

    try:
        # decode Base64
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

        certainty = "High" if confidence >= 0.7 else "Medium" if confidence >= 0.5 else "Low"

        return {
            "label": label_idx,
            "label_name": label_name,
            "confidence": round(confidence, 2),
            "certainty": certainty
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
