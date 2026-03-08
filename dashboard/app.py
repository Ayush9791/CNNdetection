from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import batch_predict_folder, load_model, predict_image

st.set_page_config(page_title="Lung Cancer CNN Dashboard", page_icon="🫁", layout="wide")

st.title("🫁 Lung Cancer Detection Dashboard (CNN)")
st.caption("Upload chest scan images and run CNN inference.")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="artifacts/lung_cancer_cnn.h5")
    img_size = st.slider("Image size", min_value=64, max_value=512, value=224, step=32)
    threshold = st.slider("Cancer threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if not Path(model_path).exists():
    st.warning("Model file not found. Train a model first or update the model path in the sidebar.")
    st.stop()

model = load_model(model_path)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Single image prediction")
    uploaded_file = st.file_uploader("Upload chest scan image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_image_path = tmp.name

        pred = predict_image(model, temp_image_path, img_size=img_size, threshold=threshold)

        st.image(temp_image_path, caption="Uploaded scan", use_container_width=True)
        st.metric("Predicted Label", pred.predicted_label.upper())
        st.metric("Cancer Probability", f"{pred.probability_cancer:.2%}")

        chart_df = pd.DataFrame(
            {
                "class": ["normal", "cancer"],
                "probability": [1 - pred.probability_cancer, pred.probability_cancer],
            }
        )
        st.bar_chart(chart_df.set_index("class"))

with col2:
    st.subheader("Batch folder prediction")
    folder_path = st.text_input("Folder path", value="")

    if st.button("Run batch prediction"):
        if not folder_path:
            st.info("Enter a folder path containing images.")
        elif not Path(folder_path).exists():
            st.error("Folder path does not exist.")
        else:
            results = batch_predict_folder(model, folder_path, img_size=img_size, threshold=threshold)
            if not results:
                st.warning("No supported image files found in folder.")
            else:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("Disclaimer: This tool is for research/education and not for clinical diagnosis.")
