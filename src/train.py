from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.model import ModelConfig, build_lung_cancer_cnn


def create_dataset(directory: str, img_size: int, batch_size: int, shuffle: bool = True):
    ds = keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def train(args: argparse.Namespace) -> None:
    config = ModelConfig(img_size=args.img_size, learning_rate=args.learning_rate)

    train_ds = create_dataset(args.train_dir, args.img_size, args.batch_size, shuffle=True)
    val_ds = create_dataset(args.val_dir, args.img_size, args.batch_size, shuffle=False)

    model = build_lung_cancer_cnn(config)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.output_model, save_best_only=True, monitor="val_auc", mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = Path(args.output_model).with_suffix(".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(metrics, indent=2))

    print(f"Training complete. Best model saved to: {args.output_model}")
    print(f"Training history saved to: {history_path}")

    if args.test_dir:
        test_ds = create_dataset(args.test_dir, args.img_size, args.batch_size, shuffle=False)
        results = model.evaluate(test_ds, return_dict=True)
        print("Test metrics:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for lung cancer detection.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-model", default="artifacts/lung_cancer_cnn.h5")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
