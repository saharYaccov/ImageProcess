# ImageProcess - CNN and Deep Learning

[GitHub Repository](https://github.com/saharYaccov/ImageProcess)

## About
This project focuses on image classification using a **Convolutional Neural Network (CNN)** implemented in Python.  
The renders were generated via **FastAPI** that I built, using images collected from Kaggle, with a total dataset size of **over 10,000 images**.

## Folder Structure
ImageProcess/
│
├── cnn_model.pth # Trained model weights
├── app.py # FastAPI app for rendering and prediction
├── index.html # Web interface to display results
├── predict.html # Web form to submit images for prediction
├── requirements.txt # Required Python packages
└── README.md # This file



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
- The model uses dropout (0.5) in fully connected layers to reduce overfitting.  
- Batch normalization is applied in all convolutional blocks to stabilize training.  
- The model is designed to run on GPU if available, otherwise CPU.  

## Running the Model
To run the FastAPI app and perform image predictions:

```bash
python app.py


You can then open index.html or predict.html in a browser to upload images and get predictions from the trained model.

Dataset
Data collected from Kaggle.

Total dataset size exceeds 10,000 images across training, validation, and test sets.

Folder structure separates AI-generated and real images for each dataset split.

Model Weights
The trained model weights are saved as cnn_model.pth and can be reloaded for inference or further training.

graphql
Copy code
