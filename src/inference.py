from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


@dataclass(frozen=True)
class PredictionResult:
    probability_cancer: float
    predicted_label: str


def load_model(model_path: str):
    return keras.models.load_model(model_path)


def preprocess_image(image_path: str, img_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr

def predict_image(model, image_path: str, img_size: int = 224) -> PredictionResult:
    image_arr = preprocess_image(image_path, img_size)

    prob_normal = float(model.predict(image_arr, verbose=0)[0][0])
    prob_cancer = 1 - prob_normal

    label = "cancer" if prob_cancer >= 0.5 else "normal"

    return PredictionResult(
        probability_cancer=prob_cancer,
        predicted_label=label
    )


