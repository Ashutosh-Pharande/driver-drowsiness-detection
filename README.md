# Driver Drowsiness Detection using CNN and OpenCV

## 📌 Overview

Driver drowsiness is one of the major causes of road accidents. This project implements a **real-time driver drowsiness detection system** using **Computer Vision and Deep Learning**. The system monitors the driver's eyes using a webcam and detects whether they are **open or closed**. If the driver's eyes remain closed for a certain duration, an **alarm alert** is triggered to warn the driver.

---

## 🚀 Features

* Real-time face and eye detection using OpenCV
* Eye state classification (Open / Closed)
* Convolutional Neural Network (CNN) for eye state prediction
* Alarm alert when drowsiness is detected
* Works with laptop webcam or external/mobile camera

---

## 🧠 Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Imutils

---

## 📂 Project Structure

driver-drowsiness-detection
│
├── detection/
│ ├── detect_realtime.py
│ └── alarm.py
│
├── training/
│ ├── model.py
│ ├── preprocess.py
│ └── train.py
│
├── scripts/
│ └── split_dataset.py
│
├── utils/
│ └── helpers.py
│
├── models/
│ └── cnn_model.h5
│
├── main.py
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

git clone https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git

cd driver-drowsiness-detection

### 2️⃣ Create Virtual Environment

python -m venv venv

### 3️⃣ Activate Virtual Environment

Windows
venv\Scripts\activate

### 4️⃣ Install Dependencies

pip install -r requirements.txt

### 5️⃣ Run the Project

python main.py

---

## 📊 Model Details

The system uses a **Convolutional Neural Network (CNN)** trained on eye images classified into two categories:

* Open Eyes
* Closed Eyes

The trained model is saved as:

models/cnn_model.h5

This model is used during real-time detection to classify the driver's eye state.

---

## ⚠️ Dataset

The dataset used for training contains eye images labeled as **open** and **closed**.

Due to size limitations, the dataset is **not included in this repository**.

---

## 🔔 How It Works

1. Webcam captures real-time video frames.
2. OpenCV detects the face and eye region.
3. The trained CNN model predicts whether the eyes are **open or closed**.
4. If eyes remain closed for multiple frames, the system detects **drowsiness**.
5. An **alarm sound** is triggered to alert the driver.

---

## 🎯 Future Improvements

* Improve model accuracy using a larger dataset
* Implement eye aspect ratio (EAR) based detection
* Deploy on embedded systems like Raspberry Pi
* Integrate with vehicle monitoring systems

---

## 👨‍💻 Author

Ashutosh Pharande
Final Year B.E. – Artificial Intelligence & Data Science
