"""
predict.py

Loads a trained model and predicts the disease class for a single leaf image,
along with a treatment recommendation.

Usage:
    python predict.py --image_path sample_leaf.jpg --model_path saved_model/crop_disease_model.h5
"""

import argparse
import json
import os

import numpy as np
from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_single_image, IMG_SIZE
from treatment_recommendations import get_recommendation


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the leaf image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .h5 model")
    parser.add_argument(
        "--class_indices_path",
        type=str,
        default="saved_model/class_indices.json",
        help="Path to a JSON file mapping class names to indices (generated during training)",
    )
    return parser.parse_args()


def load_class_names(class_indices_path):
    if os.path.exists(class_indices_path):
        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)
        # Invert {class_name: index} -> {index: class_name}
        return {v: k for k, v in class_indices.items()}
    return None


def main():
    args = parse_args()

    model = load_model(args.model_path)
    img_array = preprocess_single_image(args.image_path, img_size=IMG_SIZE)

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    class_names = load_class_names(args.class_indices_path)
    predicted_class = class_names[predicted_index] if class_names else f"class_{predicted_index}"

    recommendation = get_recommendation(predicted_class)

    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print(f"Recommendation  : {recommendation}")


if __name__ == "__main__":
    main()
