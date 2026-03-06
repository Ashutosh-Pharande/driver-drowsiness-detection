# Driver Drowsiness Detection using CNN and OpenCV

## 📌 Project Overview

Driver drowsiness is one of the leading causes of road accidents. This project implements a **real-time driver drowsiness detection system** using **Computer Vision and Deep Learning**.

The system monitors the driver's eyes using a webcam and detects whether the eyes are **open or closed**. If the driver's eyes remain closed for a continuous period, an **alarm sound is triggered** to alert the driver.

---

## 🚀 Features

* Real-time face detection using OpenCV
* Eye state classification (Open / Closed)
* Convolutional Neural Network (CNN) for eye detection
* Alarm alert when drowsiness is detected
* Works with laptop webcam or mobile camera

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
│   ├── detect_realtime.py
│   └── alarm.py
│
├── training/
│   ├── model.py
│   ├── preprocess.py
│   └── train.py
│
├── scripts/
│   └── split_dataset.py
│
├── utils/
│   └── helpers.py
│
├── models/
│   └── cnn_model.h5
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

Windows:
venv\Scripts\activate

### 4️⃣ Install Dependencies

pip install -r requirements.txt

---

## ▶️ Running the Project

Run the real-time detection system:

python main.py

The webcam will start and the system will monitor the driver's eyes.

---

## 📊 Model Details

The system uses a **Convolutional Neural Network (CNN)** trained on eye images classified into two categories:

* Open Eyes
* Closed Eyes

The trained model is saved as:

models/cnn_model.h5

This model is used for **real-time eye state prediction** during webcam detection.

---

## 📊 Dataset

The model was trained using the **Closed Eyes in the Wild (CEW) Dataset**.

This dataset contains eye images categorized into two classes:

* Open Eyes
* Closed Eyes

The dataset is widely used for **eye state classification and drowsiness detection research**.

Due to repository size limitations, the dataset is **not included in this repository**.

---

## 🔁 Reproducing the Model

To train the model yourself:

1. Download the **Closed Eyes in the Wild (CEW) Dataset**
2. Place the dataset inside the project folder as:

dataset/train/open
dataset/train/closed

3. Run the training script:

python training/train.py

---

## 🔔 How It Works

1. Webcam captures real-time video frames.
2. OpenCV detects the driver's **face and eye region**.
3. The trained **CNN model** predicts whether the eyes are **open or closed**.
4. If the eyes remain closed for multiple frames, the system detects **drowsiness**.
5. An **alarm sound** is triggered to alert the driver.

---

## 🎯 Future Improvements

* Improve model accuracy using larger datasets
* Implement Eye Aspect Ratio (EAR) based detection
* Deploy on embedded systems such as Raspberry Pi
* Integrate with vehicle safety systems

---

## 👨‍💻 Author

Ashutosh Pharande
Final Year B.E. – Artificial Intelligence & Data Science
