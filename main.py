import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

FILE_ID = "1MimIt5qq_NyzxqGoZIBakqv4JhdpkjRL"
MODEL_PATH = "brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("🧠 Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "png", "jpeg","JPG"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((256, 256))
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error("⚠️ Tumor detected")
    else:
        st.success("✅ No tumor detected")
