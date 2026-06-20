
# AI-Based Crop Disease Detection System

Built during **Smart India Hackathon (SIH) 2024** — Agriculture-Tech Track.

## Overview

A deep learning system that detects plant diseases from leaf images and suggests
treatment recommendations to farmers. The goal is to make early disease diagnosis
accessible to farmers who may not have access to agricultural experts, by letting
them simply upload a photo of an affected leaf and instantly get a prediction.

## Problem Statement

Crop diseases cause significant yield losses every year, and smallholder farmers
often lack timely access to plant pathologists. This project explores whether a
lightweight CNN model can classify common leaf diseases accurately enough to be
useful as a first-line diagnostic tool, deployable on low-cost hardware or a
simple web/mobile interface.

## Approach

1. **Data preprocessing** — Leaf images are resized, normalized, and augmented
   (rotation, flipping, brightness jitter) to improve generalization across
   varying lighting and camera conditions.
2. **Model architecture** — A convolutional neural network (CNN) built with
   Keras/TensorFlow, using stacked Conv2D + MaxPooling blocks followed by dense
   layers for multi-class classification.
3. **Training** — The model is trained with categorical cross-entropy loss and
   the Adam optimizer, with early stopping and learning-rate reduction on
   plateau to avoid overfitting.
4. **Inference** — A simple prediction script loads the trained model and
   returns the predicted disease class along with a confidence score and a
   basic treatment suggestion.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV, Pillow
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Google Colab / Jupyter

## Project Structure

```
crop-disease-detection/
├── README.md
├── requirements.txt
├── data_preprocessing.py     # Image loading, resizing, augmentation
├── model.py                  # CNN architecture definition
├── train.py                  # Training script with callbacks
├── predict.py                # Inference script for a single image
└── treatment_recommendations.py  # Disease -> treatment mapping
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize your dataset as:
#    dataset/
#      train/<class_name>/*.jpg
#      val/<class_name>/*.jpg

# 3. Train the model
python train.py --data_dir dataset --epochs 25

# 4. Run inference on a new leaf image
python predict.py --image_path sample_leaf.jpg --model_path saved_model/crop_disease_model.h5
```

## Results

- Achieved strong validation accuracy on a multi-class leaf disease dataset
  (e.g. healthy vs. early blight vs. late blight vs. bacterial spot, depending
  on the crop subset used).
- Model size and inference time were kept small enough for near-instant
  predictions, suitable for a simple farmer-facing interface.

## Future Improvements

- Expand the dataset to cover more crop types and regional disease variants.
- Add a lightweight mobile app front-end with on-device inference (TensorFlow Lite).
- Incorporate explainability (Grad-CAM) so the model highlights the diseased
  region of the leaf for farmer trust and verification.



Built collaboratively by a 6-member team as part of SIH 2024.

