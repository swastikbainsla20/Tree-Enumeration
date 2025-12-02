🌳 Tree Detection in Aerial Images & Largest Safe Region Mapping

This project implements a YOLO-based deep learning system that detects trees from aerial/drone images and computes the largest safe, tree-free region using spatial mask generation. Built on a custom dataset with strong performance metrics, it supports full training, evaluation, inference, and an interactive Streamlit interface.

🚀 Features

YOLO-based Tree vs Not-Tree detection

Spatial mask generation to compute the largest vegetation-free region

High evaluation performance with strong mAP scores

Complete training + inference pipeline

Interactive Streamlit app for live visualization

Modular and easily extendable for multi-class vegetation or GIS integration

📊 Model Performance

mAP@0.5 (overall): 0.729

Tree mAP: 0.797

Not-Tree mAP: 0.662

Best threshold: ~0.35 (optimal F1 score)

Evaluation includes PR curves, F1 curves, confidence curves, confusion matrices, class distribution, and box-geometry heatmaps.

🧠 Methodology

Model: YOLO

Optimizer: AdamW (lr=0.0008)

Augmentations: Flip, scale, HSV

Dataset: Custom aerial/drone imagery with 48k+ annotations

Post-processing: Mask extraction + largest safe-region computation

🛠️ Tech Stack

Python

PyTorch

Ultralytics YOLO

OpenCV

NumPy

Streamlit

📂 Folder Structure
├── data/             # Dataset & labels
├── models/           # Trained YOLO weights
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── mask_generator.py
│   └── visualize.py
├── results/          # Metrics & plots
└── app/              # Streamlit interface

▶️ How to Run
Install dependencies
pip install -r requirements.txt

Run inference
python src/infer.py --img path/to/image.jpg

Launch Streamlit app
streamlit run app/app.py

📌 Applications

Renewable-energy site planning

Drone-based forest monitoring

Vegetation analysis

Environmental and land-use mapping

🔮 Future Improvements

Multi-class vegetation detection

GIS integration

Real-time drone-edge inference

Larger, more diverse datasets
