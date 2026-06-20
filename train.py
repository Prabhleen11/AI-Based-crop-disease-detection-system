"""
train.py

Trains the crop disease classification CNN and saves the best model.

Usage:
    python train.py --data_dir dataset --epochs 25
"""

import argparse
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from data_preprocessing import get_data_generators, IMG_SIZE
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train crop disease detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="saved_model", help="Where to save the model")
    return parser.parse_args()


def plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    print(f"Saved training history plot to {output_dir}/training_history.png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_gen, val_gen = get_data_generators(args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size)
    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes: {train_gen.class_indices}")

    model = build_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
    model.summary()

    checkpoint_path = os.path.join(args.output_dir, "crop_disease_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, args.output_dir)
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
