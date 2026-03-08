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
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(model, image_path: str, img_size: int = 224, threshold: float = 0.5) -> PredictionResult:
    image_arr = preprocess_image(image_path, img_size)
    prob_cancer = float(model.predict(image_arr, verbose=0)[0][0])
    label = "cancer" if prob_cancer >= threshold else "normal"
    return PredictionResult(probability_cancer=prob_cancer, predicted_label=label)


def batch_predict_folder(model, folder_path: str, img_size: int = 224, threshold: float = 0.5):
    results = []
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    for path in sorted(Path(folder_path).iterdir()):
        if path.suffix.lower() not in valid_ext:
            continue

        pred = predict_image(model, str(path), img_size=img_size, threshold=threshold)
        results.append(
            {
                "file": path.name,
                "predicted_label": pred.predicted_label,
                "probability_cancer": pred.probability_cancer,
                "probability_normal": 1.0 - pred.probability_cancer,
            }
        )

    return results
