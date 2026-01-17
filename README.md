# Pneumonia Detection API (Chest X-ray)

This repository contains a small deep learning project that classifies chest X-ray images as:

- **NORMAL**
- **PNEUMONIA**

It uses **transfer learning with ResNet18**, trained on a chest X-ray dataset, and exposes a simple **FastAPI** endpoint for inference.  
The app is containerized using **Docker** and can be deployed to cloud platforms like **Render**.

---

## 📌 Dataset

The dataset used is **Chest X-Ray Images (Pneumonia)**.

> ⚠️ The dataset is **not included** in this repository due to size constraints.  
> You must download it manually and place it in the `data/` folder with the structure below.

The dataset can be downloaded from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
---

## 🚀 Project Structure

```bash
├── data/
│ ├── train/
│ ├── val/
│ └── test/
├── artifacts/
│ └── model.pth
├── train.py
├── evaluate.py
├── predict.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 🧠 Model

- **Base model:** ResNet18
- **Pretrained:** ImageNet
- **Output:** Binary classification
- **Loss:** `BCEWithLogitsLoss`
- **Threshold:** `0.7` (optimized for high recall)

---

## 🧰 Requirements

This project uses **uv** for dependency management.

---

## 📌 Installation

Install dependencies:

```bash
uv install
```

---

## 🏋️ Training

Train the model:

```bash
python train.py
```

This script will:
- Train the model
- Save the weights to model.pth

---

## 📊 Evaluation

Run evaluation:

```bash
python evaluate.py
```

This script prints:
- Confusion matrix
- Precision and recall
- Threshold comparison

---

## 🧪 Prediction API

Start the FastAPI server:

```bash
uvicorn predict:app --reload
```

Then open: http://localhost:8000/docs and upload an X-ray image to receive a prediction.

---

## 🐳 Docker

Build the image: 

```bash
docker build -t pneumonia-api .
```

Run the container:

```bash
docker run -p 8000:8000 pneumonia-api
```

- Example of docker image running:

<img width="535" height="216" alt="image" src="https://github.com/user-attachments/assets/de83c348-596d-4b65-999f-0109f65f1df3" />

---

## 🚀 Deployment on Render

The docker image was deployed in Render. 
Use the URL below to upload an example X-ray image from the Kaggle dataset mentioned above to test the app.

**URL:** https://ml-zoomcamp-2025-capstone-project.onrender.com/docs

---

## 🧾 License

This project is for educational use only.
It is not a medical device.

Do not use it for real-world diagnosis.

---




