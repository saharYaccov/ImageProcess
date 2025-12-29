# ImageProcess - CNN and Deep Learning 🖼️🤖

[GitHub Repository](https://github.com/saharYaccov/ImageProcess)  

## 🌐 **Live Demo:** 

👉 [Try ImageProcess Online – Frontend (GitHub Pages, html) + Backend (Render)](https://saharyaccov.github.io/ImageProcess/)

👉 [Try ImageProcess Online - Hugging Face](https://huggingface.co/spaces/sahar-yaccov/imagePrediction)  

🚀 **Render Dashboard:** [View deployment events](https://dashboard.render.com/web/srv-d56qr4mr433s73eb55d0/events)
Deployment & Integration 🚀

The project is deployed on Render, where the FastAPI backend handles GET and POST requests for image predictions.
The frontend, hosted on GitHub Pages, communicates with the backend to send images for prediction and display the results in real-time.
This setup ensures a seamless end-to-end pipeline from image upload to AI-generated classification without requiring local execution. 🌐⚡

## About
This project focuses on **image classification** using a **Convolutional Neural Network (CNN)** implemented in Python with **PyTorch**.  
The system provides **end-to-end inference** through a **FastAPI** backend and a web-based frontend. 
The model has been trained and tested on 1,500+ {⏱️ Total Time Running : 10'  - 15'  Min } images, achieving high accuracy and reliable predictions. 📊

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT8F81dmy782i-FrHwcDy8maYnLUpObPsnhJA&s" 
     alt="Deep Learning" 
     style="width: 80%; max-width: 800px; height: auto; display: block; margin: 20px auto;">

*Illustration of deep learning concept*

---

## Folder Structure 📁

| Folder / File        | Description |
|---------------------|-------------|
| `cnn_model.pth`      | Trained CNN model weights 🏋️‍♂️ |
| `app.py`             | FastAPI application for model inference ⚡ |
| `index.html`         | Main web interface 🌐 |
| `predict.html`       | Image upload and prediction form 📸 |
| `requirements.txt`   | Python dependencies 📦 |
| `README.md`          | Project documentation 📖 |

---

## Model Overview 🧠
The model is a **Convolutional Neural Network (CNN)** designed to classify images into **two classes**:  
**AI-generated images** (`ai_image`) vs **Real images** (`real_image`). ✅

- **Test Accuracy:** ~ 70% 🎯  
- **Validation Accuracy:** ~ 70% 🎯

The architecture uses **4 convolutional blocks** followed by fully connected layers with **dropout** to reduce overfitting.

![CNN Example](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)
*Typical CNN architecture diagram*

---

## Architecture Details 🏗️

- **Input:** RGB images resized to **224 × 224** 📐

### Convolutional Feature Extractor
The network consists of **four convolutional blocks**:


# CNN Model: AI vs Real Image Classifier

## Architecture Overview
- **Architecture:** 4 Convolutional Blocks + Classifier (Fully Connected Layers)  
- **Input Image Size:** 224×224 RGB  
- **Dataset Size (example run):** 500+ images  

---

## Convolutional Blocks

## Architecture Overview
- **Architecture:** 6 Convolutional Blocks + Classifier (Fully Connected Layers)  
- **Input Image Size:** 224×224 RGB  
- **Dataset Size:** 420 training images, 90 test images  

---

1. **Conv Block 1**  
   - `Conv2d`: 3 → 16 channels, kernel size 3×3, stride 1, padding 1  
   - `BatchNorm2d(16)`  
   - `ReLU` activation  
   - `MaxPool2d`: 2×2, stride 2  

2. **Conv Block 2**  
   - `Conv2d`: 16 → 32 channels, kernel size 3×3, stride 1, padding 1  
   - `BatchNorm2d(32)`  
   - `ReLU` activation  
   - `MaxPool2d`: 2×2, stride 2  

3. **Conv Block 3**  
   - `Conv2d`: 32 → 64 channels, kernel size 3×3, stride 1, padding 1  
   - `BatchNorm2d(64)`  
   - `ReLU` activation  
   - `MaxPool2d`: 2×2, stride 2  

4. **Conv Block 4**  
   - `Conv2d`: 64 → 128 channels, kernel size 3×3, stride 1, padding 1  
   - `BatchNorm2d(128)`  
   - `ReLU` activation  
   - `MaxPool2d`: 2×2, stride 2  

5. **Conv Block 5**  
   - `Conv2d`: 128 → 256 channels, kernel size 3×3, stride 1, padding 1  
   - `BatchNorm2d(256)`  
   - `ReLU` activation  
   - `MaxPool2d`: 2×2, stride 2  

6. **Conv Block 6**  
   - `Conv2d`: 256 → 512 channels, kernel size 3×3, stride 1, padding 1  
   - `BatchNorm2d(512)`  
   - `ReLU` activation  
   - `MaxPool2d`: 2×2, stride 2  

---

## Classifier
- `Flatten(start_dim=1)`  
- `Dropout(p=0.5)`  
- `Linear`: 4608 → 512  
- `ReLU` activation  
- `Dropout(p=0.5)`  
- `Linear`: 512 → 2 (output classes: AI-generated / Real)  

---

## Training Details
- Number of epochs: **18**  
- Loss progression:
  - Epoch 1: 0.6448  
  - Epoch 2: 0.5409  
  - Epoch 3: 0.5030  
  - Epoch 4: 0.4765  
  - Epoch 5: 0.4582  
  - Epoch 6: 0.4202  
  - Epoch 7: 0.4091  
  - Epoch 8: 0.3731  
  - Epoch 9: 0.3438  
  - Epoch 10: 0.3149  
  - Epoch 11: 0.3038  
  - Epoch 12: 0.2823  
  - Epoch 13: 0.2514  
  - Epoch 14: 0.2298  
  - Epoch 15: 0.2136  
  - Epoch 16: 0.1895  
  - Epoch 17: 0.1834  
  - Epoch 18: 0.1573  
- Test Accuracy: **85.27%**  
- Validation Accuracy: **85.35%**  
- Dataset Size: 5600 training images, 1290 test images (~5000+ total)  
- Input Image Size: 224×224 RGB  
- Training device: Apple Silicon GPU (MPS)  
- CPU threads used: 10
- 
**Classes:**  
- `0` → `ai_image` 🤖  
- `1` → `real_image` 🏞️


![Classification](https://www.baeldung.com/wp-content/uploads/sites/4/2023/04/Fig-4-1-scaled.jpg)
*Example of CNN feature maps*

---

## Training Configuration ⚙️
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Number of Epochs:** ~5  
- **Hardware:** GPU if available, otherwise CPU 💻

---

## Additional Notes 📝
- Dropout reduces overfitting and improves generalization  
- Moderate architecture chosen to balance capacity and simplicity  
- Exposed via REST API for **real-time inference** 🌐

---

## Running the Model ▶️
To run the FastAPI app locally:

```bash
python app.py
