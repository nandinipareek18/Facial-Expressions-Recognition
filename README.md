# Facial Expression Recognition 🎭

A deep learning–based Facial Expression Recognition system that detects human emotions in real time using a webcam.  
The model is trained on the **FER2013 dataset** and uses a **CNN (Little VGG-style architecture)** with TensorFlow/Keras and OpenCV.

---

## 🔥 Features
- Real-time facial emotion detection using webcam
- Custom CNN architecture optimized for grayscale facial images
- Face detection using Haar Cascade
- Trained on FER2013 dataset
- Robust to lighting and facial variations
- Supports 5 emotions:
  - Angry
  - Happy
  - Neutral
  - Sad
  - Surprise

---

## 🧠 Model Architecture
- Input size: **48 × 48 (Grayscale)**
- Convolutional Blocks with:
  - Conv2D
  - ELU activation
  - Batch Normalization
  - MaxPooling
  - Dropout
- Fully Connected layers
- Softmax output layer

Total Parameters: ~1.3M

---


