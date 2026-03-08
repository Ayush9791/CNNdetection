from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.inference import load_model, predict_image

st.set_page_config(
    page_title="Lung Cancer Detection",
    layout="centered"
)

st.title("Lung Cancer Detection")
st.caption("Upload a chest scan image to analyze using the trained CNN model.")

MODEL_PATH = "artifacts/lung_cancer_cnn.h5"
IMG_SIZE = 224


if not Path(MODEL_PATH).exists():
    st.error("Model file not found. Train the model first.")
    st.stop()


@st.cache_resource
def load_cached_model(path):
    return load_model(path)


model = load_cached_model(MODEL_PATH)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload chest scan image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_file:

    suffix = Path(uploaded_file.name).suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    pred = predict_image(model, image_path, img_size=IMG_SIZE)

    st.image(image_path, caption="Uploaded Scan", use_container_width=True)

    st.markdown("### Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Diagnosis",
            pred.predicted_label.upper()
        )

    with col2:
        st.metric(
            "Cancer Probability",
            f"{pred.probability_cancer:.2%}"
        )

    chart_df = pd.DataFrame(
        {
            "class": ["Normal", "Cancer"],
            "probability": [
                1 - pred.probability_cancer,
                pred.probability_cancer
            ]
        }
    )

    st.markdown("### Probability Distribution")
    st.bar_chart(chart_df.set_index("class"))

st.markdown("---")
st.caption("Research tool only. Not intended for medical diagnosis.")