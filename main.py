import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "brain_tumor_model.h5"
IMG_SIZE = (224, 224)  

@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_trained_model()

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload the image : ")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction.shape[1] == 1:  
        prob = prediction[0][0]
        label = "Tumor" if prob > 0.5 else "No Tumor"
        st.write(f"### Prediction: **{label}** (prob={prob:.2f})")
    else:
        class_idx = np.argmax(prediction)
        st.write(f"### Prediction: Class {class_idx} - Probabilities: {prediction}")
