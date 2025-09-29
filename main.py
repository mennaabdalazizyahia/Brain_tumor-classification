import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time

FILE_ID = "1MimIt5qq_NyzxqGoZIBakqv4JhdpkjRL"
MODEL_PATH = "brain_tumor_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('📥 Downloading model...'):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    with st.spinner('🔄 Loading model...'):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None

st.title("🧠 Brain Tumor Detection System")

model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg","JPG","PNG"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(image, use_column_width=True)
        
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.5:
            st.error(f"🚨 Tumor Detected ({prediction*100:.1f}%)")
        else:
            st.success(f"✅ No Tumor ({(1-prediction)*100:.1f}%)")
else:
    st.error("""
    **the model is not downloaded:**
    1.  try this version of tensorflow : pip install tensorflow==2.13.0`
    """)
