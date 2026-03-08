# Lung Cancer Detection with CNN + Dashboard

This project provides:

1. A **Convolutional Neural Network (CNN)** training pipeline for binary lung cancer image classification.
2. A **dashboard app** (Streamlit) for end-user prediction and model monitoring.

> Expected classes:
> - `normal`
> - `cancer`

## Project structure

- `src/model.py` – CNN architecture and helpers.
- `src/train.py` – Training pipeline using image folders.
- `src/inference.py` – Single image prediction helpers.
- `dashboard/app.py` – Streamlit dashboard for predictions and visualization.
- `requirements.txt` – Python dependencies.

## Dataset format

Organize your dataset as:

```text
data/
  train/
    normal/
    cancer/
  val/
    normal/
    cancer/
  test/
    normal/
    cancer/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model

```bash
python -m src.train \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 20 \
  --batch-size 32 \
  --img-size 224 \
  --output-model artifacts/lung_cancer_cnn.h5
```

## Evaluate model

```bash
python -m src.train \
  --train-dir data/train \
  --val-dir data/val \
  --test-dir data/test \
  --epochs 20 \
  --output-model artifacts/lung_cancer_cnn.h5
```

## Run dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard features:
- Upload chest scan image for prediction.
- Probability chart for class confidence.
- Sidebar settings for image size and threshold.
- Batch prediction from a folder.

## Notes

- This project is for educational/research workflows; not a medical device.
- Use clinically validated datasets and evaluation protocols before real-world use.
