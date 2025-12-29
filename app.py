import os
import io
import base64
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# משיכת המפתח ממשתני סביבה - ב-Render תגדיר תחת Env Vars את API_KEY
API_KEY = os.getenv("API_KEY", "default_secret_if_not_found")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ליתר ביטחון בשלב הבדיקות, עדיף להחליף לדומיין של הגיטהאב שלך בהמשך
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "Image Process API is running", "status": "active"}

# הגדרת המודל (ללא שינוי בארכיטקטורה)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = nn.Sequential(
    # Conv Block 1
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Conv Block 2
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Conv Block 3
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Conv Block 4
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Conv Block 5
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Conv Block 6
    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Classifier
    nn.Flatten(start_dim=1),
    nn.Dropout(0.5),
    nn.Linear(4608, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
).to(device)
# טעינת המשקולות
try:
    if os.path.exists("cnn_model.pth"):
        model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
        model.eval()
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Warning: cnn_model.pth not found.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/predict")
async def predict(request: ImageRequest, x_api_key: str = Header(None)):
    # אימות מפתח בצורה בטוחה
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        # ניקוי הסטרינג למקרה שנשלח עם ה-prefix של ה-data:image
        encoded_data = request.image_base64.split(",")[-1]
        image_data = base64.b64decode(encoded_data)
        
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        x = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            y = model(x)
            prob = torch.softmax(y, dim=1)
            label_idx = int(torch.argmax(prob))
            confidence = float(torch.max(prob))

        class_names = ["AI-generated", "Real"]
        print({
            "🏷️ label_name": class_names[label_idx],
            "📊 confidence": round(confidence, 2),
            "✅ status": "success"
        })
        return {
            "label_name": class_names[label_idx],
            "confidence": round(confidence, 2),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.get("/health")
def health():
    return {"status": "online"}
