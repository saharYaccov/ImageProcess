# ImageProcess - CNN and Deep Learning

[GitHub Repository](https://github.com/saharYaccov/ImageProcess)  

## **Live Demo:** [Try ImageProcess online](https://saharyaccov.github.io/ImageProcess/)


## About
This project focuses on image classification using a **Convolutional Neural Network (CNN)** implemented in Python.  
The renders were generated via **FastAPI** that I built, using images collected from Kaggle, with a total dataset size of **over 10,000 images**.

## Folder Structure

| Folder / File        | Description |
|---------------------|-------------|
| `cnn_model.pth`      | Trained model weights |
| `app.py`             | FastAPI app for rendering and prediction |
| `index.html`         | Web interface to display results |
| `predict.html`       | Web form to submit images for prediction |
| `requirements.txt`   | Required Python packages |
| `README.md`          | Project documentation (this file) |

## Model Overview
The model is a **Convolutional Neural Network (CNN)** designed to classify images into 2 categories: AI-generated vs real.  
It is composed of **4 convolutional blocks** followed by fully connected layers, with dropout and batch normalization to improve generalization.  

**Architecture Details:**
- **Input:** RGB images of size 224x224  
- **Convolutional Blocks:**  
  1. Conv2d → BatchNorm → ReLU → MaxPool2d  
  2. Conv2d → BatchNorm → ReLU → MaxPool2d  
  3. Conv2d → BatchNorm → ReLU → MaxPool2d  
  4. Conv2d → BatchNorm → ReLU → MaxPool2d  
- **Flatten Layer:** Converts 3D feature maps to 1D vector  
- **Fully Connected Layers:**  
  - Linear → ReLU/Sigmoid → Dropout  
  - Linear → 2 output classes (logits)  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Number of Epochs:** Typically 5 for demonstration  

**Additional Notes:**
- Dropout (0.5) is applied in fully connected layers to reduce overfitting.  
- Batch normalization stabilizes training across convolutional blocks.  
- The model runs on GPU if available, otherwise CPU.  

## Running the Model
To run the FastAPI app and perform image predictions:

```bash
python app.py
