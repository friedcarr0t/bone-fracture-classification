# Bone Fracture Classification System

This repository contains a deep learning–based **bone fracture classification system** built using a **Fusion CNN–Transformer architecture**. The project focuses on classifying **10 different types of bone fractures** from X-ray images and is complemented by a **web-based prediction interface**.

This project is intended for **educational, experimental, and portfolio purposes** in computer vision and medical image analysis. It is **not intended for clinical or diagnostic use**.

---

## Project Overview

* Task: Multi-class bone fracture classification from X-ray images
* Number of classes: 10 fracture categories
* Model architecture: Fusion CNN–Transformer (EfficientNetB3 + attention mechanism)
* Deployment: Streamlit web application for real-time inference

---

## Dataset

* Total images: 1,129 X-ray images
* Modality: Bone X-ray images
* Classes: 10 fracture types
* Dataset structure follows a standard directory-based format used for image classification.

> Note: The dataset is included inside the repository under the appropriate directory. The model performance is optimized for this dataset and may not generalize to real-world clinical data with different imaging conditions.

---

## Data Preprocessing

The following preprocessing steps are applied during training:

* Image resizing to 224 × 224
* Pixel value normalization
* Data augmentation (rotation, flipping, zooming)

All preprocessing logic is implemented directly inside the training notebook.

---

## Model Architecture

The model uses a **Fusion CNN–Transformer** approach:

* Backbone: EfficientNetB3 (pretrained)
* Feature extraction using CNN layers
* Attention mechanism to capture global contextual information
* Fully connected layers for classification
* Output layer with Softmax activation for 10-class prediction

Training configuration:

```
Loss Function : Categorical Cross-Entropy
Optimizer     : Adam
Metrics       : Accuracy, F1-Score
```

---

## Model Training

Model training is performed using the notebook:

```
model_train.ipynb
```

Training workflow:

1. Load dataset from directory
2. Apply preprocessing and augmentation
3. Build Fusion CNN–Transformer model
4. Train using training and validation data
5. Evaluate performance
6. Save trained model weights

---

## Model Performance

Evaluation on the test set:

* Accuracy: **88.57%**
* F1-Score: **88.39%**

> Performance may vary when tested on external or real-world medical datasets.

---

## Web Application

A web-based inference interface is implemented using **Streamlit**.

Features:

* Upload X-ray image
* Real-time fracture classification
* Probability distribution visualization for all classes

To run the web app locally:

```bash
streamlit run app.py
```

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
venv\\Scripts\\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Tech Stack

* Python
* TensorFlow & Keras
* Streamlit
* OpenCV
* NumPy, Pandas, Scikit-learn

---

## Disclaimer

This project is for academic and research purposes only.
It must not be used for real medical diagnosis or clinical decision-making.
