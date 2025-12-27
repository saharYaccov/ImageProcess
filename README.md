# ImageProcess - CNN and Deep Learning 🖼️🤖

[GitHub Repository](https://github.com/saharYaccov/ImageProcess)  

## 🌐 **Live Demo:** [Try ImageProcess online](https://saharyaccov.github.io/ImageProcess/)  

🚀 **Render Dashboard:** [View deployment events](https://dashboard.render.com/web/srv-d56qr4mr433s73eb55d0/events)

## About
This project focuses on **image classification** using a **Convolutional Neural Network (CNN)** implemented in Python with **PyTorch**.  
The system provides **end-to-end inference** through a **FastAPI** backend and a web-based frontend.  
The model was trained on images collected from **Kaggle**, with a total dataset size of **10,000+ images**. 📊

![Deep Learning](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/90650dnn2.webp)
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

- **Test Accuracy:** 84.62% 🎯  
- **Validation Accuracy:** 90.91% 🎯

The architecture uses **4 convolutional blocks** followed by fully connected layers with **dropout** to reduce overfitting.

![CNN Example](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)
*Typical CNN architecture diagram*

---

## Architecture Details 🏗️

- **Input:** RGB images resized to **224 × 224** 📐

### Convolutional Feature Extractor
The network consists of **four convolutional blocks**:

1. **Conv Block 1**  
   - Conv2d (3 → 16, kernel size 3×3, padding 1)  
   - ReLU  
   - MaxPool2d (2×2)  
   ![Conv Block](https://upload.wikimedia.org/wikipedia/commons/2/22/Convolutional_Neural_Network_%28CNN%29.png)

2. **Conv Block 2**  
   - Conv2d (16 → 32, kernel size 3×3, padding 1)  
   - ReLU  
   - MaxPool2d (2×2)

3. **Conv Block 3**  
   - Conv2d (32 → 64, kernel size 3×3, padding 1)  
   - ReLU  
   - MaxPool2d (2×2)

4. **Conv Block 4**  
   - Conv2d (64 → 128, kernel size 3×3, padding 1)  
   - ReLU  
   - MaxPool2d (2×2)

---

### Classification Head
- **Flatten Layer:** Converts 3D feature maps into 1D vector  
- **Dropout (0.5)** applied for regularization 🛡️  
- **Fully Connected Layer:**  
  - Linear (25088 → 128)  
  - ReLU  
  - Dropout (0.5)  
- **Output Layer:**  
  - Linear (128 → 2) producing logits for **binary classification**  

**Classes:**  
- `0` → `ai_image` 🤖  
- `1` → `real_image` 🏞️

![Classification](data:image/png;base64,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)
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
