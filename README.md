# Bone Fracture Classification

This repository contains a deep learning project for **bone fracture classification from X-ray images** using a Convolutional Neural Network (CNN).  
The objective is to classify X-ray images into **fractured** and **not fractured** categories.

This project is intended for **educational and experimental purposes** in computer vision and medical image analysis.

---

## Environment Setup
Clone the repository:   
```bash
git clone https://github.com/friedcarr0t/bone-fracture-classification.git
cd bone-fracture-classification
```
Create and activate a virtual environment:   
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```
Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras opencv-python
```

---

## Dataset Preprocessing
Preprocessing steps applied during training include:
- Image resizing to a fixed resolution (e.g. 224x224)
- Pixel normalization
- Optional data augmentation (rotation, flipping, zoom)
All preprocessing logic is implemented directly inside the training notebook.

---

## Model Architecture
The model is a CNN-based binary classifier with the following target labels:   
```bash
0 -> Not Fractured
1 -> Fractured
```
General architecture components:
- Convolutional layers
- Pooling layers
- Fully connected (Dense) layers
- Sigmoid activation for binary classification

Training configuration:
```bash
Loss Function : Binary Cross-Entropy
Optimizer     : Adam
Metrics       : Accuracy
```

---

## Model Training
Model training is performed using: `model_train.ipnyb`   
Training workflow:
```bash
1. Load dataset from directory
2. Apply preprocessing and augmentation
3. Compile CNN model
4. Train using training and validation data
5. Save the trained model
```

---

## Disclaimer
This project is for academic and research purposes only.
It must not be used for real medical diagnosis or clinical decisions.


