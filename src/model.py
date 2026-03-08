from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class ModelConfig:
    img_size: int = 224
    num_classes: int = 1
    learning_rate: float = 1e-3


def build_lung_cancer_cnn(config: ModelConfig) -> keras.Model:
    """Build a compact CNN for binary lung cancer detection."""
    inputs = keras.Input(shape=(config.img_size, config.img_size, 3), name="image")

    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="cancer_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lung_cancer_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
