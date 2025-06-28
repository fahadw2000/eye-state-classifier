# 👁️ Eye State Classifier using MobileNet

This project detects whether eyes are open or closed using a deep learning model built on top of MobileNet.

## 🧠 Overview
- Classifies eye states (open/closed) using real-time webcam input or datasets
- Trained using custom cropped RGB eye images
- Visualizes predictions in a 3x3 OpenCV grid

## 🏗️ Model Architecture
- Base: MobileNet (ImageNet pretrained)
- Custom top layers: Conv2D, Dense, Dropout
- Binary sigmoid classifier

## 📊 Evaluation
- Metrics: Accuracy, Confusion Matrix, ROC Curve
- >90% test accuracy with good generalization

## 🖥️ Technologies Used
`Python`, `TensorFlow`, `Keras`, `OpenCV`, `scikit-learn`, `matplotlib`, `seaborn`

## 🚀 Run
```bash
python Eye_classification_model.py
