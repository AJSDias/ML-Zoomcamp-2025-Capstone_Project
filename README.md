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

---

## 🚀 Project Structure
.
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
