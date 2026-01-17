# Pneumonia Detection API (Chest X-ray)

This repository contains a small deep learning project that classifies chest X-ray images as:

- **NORMAL**
- **PNEUMONIA**

It uses **transfer learning with ResNet18**, trained on a chest X-ray dataset, and exposes a simple **FastAPI** endpoint for inference.  
The app is containerized using **Docker** and can be deployed to cloud platforms like **Render**.

---

## 🚀 Project Structure

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

