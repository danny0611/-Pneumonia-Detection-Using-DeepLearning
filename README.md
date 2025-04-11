# 🫁 Pneumonia Detection Using Deep Learning

## 📌 Overview
This project focuses on detecting pneumonia from chest X-ray images using an 8-layer Convolutional Neural Network (CNN). Achieving a **96% accuracy** on validation data, the model leverages modern deep learning techniques like **Batch Normalization** and **Dropout** to improve generalization and reduce overfitting.

---

## 📂 Dataset
The dataset used in this project is sourced from Kaggle:

🔗 [Pneumonia Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset includes labeled chest X-ray images categorized into:
- **Normal**
- **Pneumonia**

---

## 🛠️ Technologies Used

- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Image Processing**: OpenCV  
- **Programming Language**: Python  
- **Model Optimization**: Batch Normalization, Dropout  
- **Data Augmentation**: Keras ImageDataGenerator

---

## 🚀 Features

- ✔️ Custom-built **8-layer CNN** architecture for pneumonia classification  
- ✔️ Achieves **96% validation accuracy**  
- ✔️ Utilizes **data augmentation** for better model generalization  
- ✔️ Supports **real-time prediction** on X-ray images  

---

## 🧠 Model Architecture

The CNN model consists of:
- Multiple convolutional layers with ReLU activation  
- MaxPooling layers to reduce spatial dimensions  
- Batch Normalization and Dropout layers for regularization  
- Fully connected dense layers  
- Output layer with sigmoid activation for binary classification  

---

## 🖼️ Sample Prediction

You can test the model on custom chest X-ray images using the provided script:
```bash
python predict.py --image path_to_image.jpg
