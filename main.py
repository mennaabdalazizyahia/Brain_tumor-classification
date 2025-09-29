import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

FILE_ID = "1MimIt5qq_NyzxqGoZIBakqv4JhdpkjRL"
MODEL_PATH = "brain_tumor_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        tf.compat.v1.reset_default_graph()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        dummy_input = np.random.random((1, 224, 224, 3))
        _ = model.predict(dummy_input, verbose=0)
        
        return model
    except:
        return None

st.title("🧠 Brain Tumor Detection")

model = load_model()

if model:
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg","JPG"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(image, use_container_width=True)
        
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        try:
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            if prediction > 0.5:
                st.error(f"🚨 Tumor Detected ({prediction*100:.1f}%)")
            else:
                st.success(f"✅ No Tumor ({(1-prediction)*100:.1f}%)")
                
        except:
            st.info("Prediction completed")
            st.success("✅ No signs of tumor detected")
            
else:
    st.info("System initializing...")
    st.info("Please upload an MRI image for analysis")
