# 😐 Facial Expression Recognition using Keras

A deep learning project to recognize facial expressions using a Convolutional Neural Network (CNN) built from scratch with Keras.

---

## 📘 Overview

This project builds and trains a CNN model using the FER-2013 dataset to classify facial expressions into one of seven emotions:

- 0 = Angry  
- 1 = Disgust  
- 2 = Fear  
- 3 = Happy  
- 4 = Sad  
- 5 = Surprise  
- 6 = Neutral  

The input images are 48x48 grayscale images of faces. OpenCV is used to detect faces and draw bounding boxes around them for real-time emotion detection.

---

## 🚀 Features

- 🧠 Trained from scratch using **Keras**
- 📦 Uses the **FER-2013** dataset
- 📹 Real-time emotion detection via webcam using **OpenCV**
- 🌐 Option to deploy model via a simple web interface
- 💾 CNN model saved and reusable for inference

---

## 🧠 Model Architecture

- Convolutional Layers with ReLU activations
- MaxPooling Layers
- Dropout for regularization
- Dense layers for classification
- Softmax for final prediction

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

Make sure you have OpenCV installed:

```bash
pip install opencv-python
```

---

## 🏁 Getting Started

### To train the model:

```bash
python train_model.py
```

### To test on an image or video:

```bash
python real_time_prediction.py
```

---

## 📂 Dataset

The model is trained on [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), which contains over 35,000 labeled 48x48 grayscale images across 7 emotion categories.

---

## 📊 Results

- Achieved significant accuracy in recognizing common facial expressions.
- Real-time performance with bounding boxes and emotion labels on detected faces.

---

## 🤖 Technologies Used

- Python
- Keras (TensorFlow backend)
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- Flask / Streamlit (optional for web deployment)

---
