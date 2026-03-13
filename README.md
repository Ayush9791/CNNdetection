# Lung Detection Model using CNN

**Author:** Ayush Deep  
**Dataset:** https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

---

## Overview

This project implements a **Convolutional Neural Network (CNN)** to detect lung cancer from **CT scan images**. The model analyzes chest CT images and classifies them into different categories of lung conditions.

Deep learning models such as CNNs are highly effective in **medical image analysis** because they automatically learn important features like edges, textures, and shapes that indicate abnormalities.

The goal of this project is to demonstrate how CNNs can assist in **early lung cancer detection**, helping doctors identify potential issues faster.

---

## Dataset

The dataset used for training and testing is the **Chest CT-Scan Images Dataset** available on Kaggle.

Dataset Link:  
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

### Classes in Dataset

The dataset contains CT scan images divided into four categories:

- **Adenocarcinoma** – A common type of lung cancer.
- **Large Cell Carcinoma** – A fast-growing cancer type.
- **Normal** – Healthy lung CT scans.
- **Squamous Cell Carcinoma** – Another type of lung cancer.

These labeled images allow the CNN to learn patterns associated with each class.

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Streamlit  
- Convolutional Neural Networks (CNN)

---

## Model Architecture

The CNN model consists of several layers used to extract and learn features from CT scan images:

1. **Convolutional Layers** – Detect patterns such as edges and textures.
2. **ReLU Activation Function** – Adds non-linearity to the model.
3. **Max Pooling Layers** – Reduce image dimensions while preserving important features.
4. **Flatten Layer** – Converts feature maps into a one-dimensional vector.
5. **Dense (Fully Connected) Layers** – Perform classification.
6. **Output Layer (Softmax/Sigmoid)** – Generates probability scores for each class.

---

## Model Training

Before training, the images go through preprocessing steps such as:

- Image resizing  
- Normalization  
- Splitting dataset into training and validation sets  

### Training Parameters

- **Optimizer:** Adam  
- **Loss Function:** Categorical/Binary Crossentropy  
- **Epochs:** 20  
- **Batch Size:** Configurable  

The model learns to classify CT images by minimizing prediction error during training.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/lung-detection-cnn.git
cd lung-detection-cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The Streamlit interface allows users to **upload CT scan images and receive predictions from the trained CNN model**.

---

## Results

The CNN model successfully learns patterns from CT scan images and predicts lung conditions with promising accuracy.

This project demonstrates the potential of **deep learning in medical image classification**.

---

## Future Improvements

Possible improvements include:

- Increasing dataset size
- Applying **data augmentation**
- Using **transfer learning models** such as ResNet or EfficientNet
- Improving model accuracy
- Deploying the model on cloud platforms

---

## Disclaimer

This project is intended for **educational and research purposes only**.  
It should not be used as a substitute for professional medical diagnosis.