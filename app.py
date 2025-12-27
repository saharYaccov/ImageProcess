from fastapi import FastAPI
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
    allow_origins=["*"],  # אפשר לשנות ל־domain ספציפי
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
# Model Definition
# -------------------------------
import torch.nn as nn

model = nn.Sequential(
    # -------------------------------
    # 1st Convolutional Block
    # -------------------------------
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

    # -------------------------------
    # 2nd Convolutional Block
    # -------------------------------
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

    # -------------------------------
    # 3rd Convolutional Block
    # -------------------------------
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

    # -------------------------------
    # 4th Convolutional Block
    # -------------------------------
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

    # -------------------------------
    # Fully Connected Layers
    # -------------------------------
    nn.Flatten(start_dim=1),
    nn.Dropout(0.5),
    nn.Linear(128 * 14 * 14, 128),  # Assuming input 224x224 -> after 4 pools: 14x14
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
).to(device)

# טען את המודל ששמרת
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
# Endpoint לחיזוי
# -------------------------------
@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # decode Base64
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        x = val_transform(image).unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            y = model(x)
            prob = torch.softmax(y, dim=1)

        return {
            "label": int(torch.argmax(prob)),
            "confidence": float(torch.max(prob))
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
