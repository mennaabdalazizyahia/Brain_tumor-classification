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
        with st.spinner('Downloading model... (This may take a few minutes)'):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    with st.spinner('🔄 Loading model...'):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None


st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection System")
st.markdown("---")

model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully! Ready for predictions.")
    
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image", 
        type=["jpg", "png", "jpeg", "JPG"],
        help="Upload a clear MRI scan image for classification: "
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Original MRI Scan", use_column_width=True)
            
            st.info(f"Image size: {image.size} | Format: {image.format}")
        
        with col2:
            st.subheader("classification Results")
            
            processed_image = image.resize((224, 224))
            img_array = np.array(processed_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner('classifying MRI scan...'):
                try:
                    start_time = time.time()
                    prediction = model.predict(img_array, verbose=0)
                    processing_time = time.time() - start_time
                    
                    tumor_probability = float(prediction[0][0])
                    
                    st.markdown("### Diagnosis Results:")
                    
                    if tumor_probability > 0.5:
                        st.error(f"**🚨 TUMOR DETECTED**")
                        st.write(f"**Confidence Level:** {tumor_probability*100:.2f}%")
                    else:
                        st.success(f"**✅ NO TUMOR DETECTED**")
                        st.write(f"**Confidence Level:** {(1-tumor_probability)*100:.2f}%")
                    
                    st.markdown("### 📈 Confidence Meter:")
                    if tumor_probability > 0.5:
                        progress_bar = st.progress(tumor_probability)
                        st.write(f"Tumor likelihood: {tumor_probability*100:.1f}%")
                    else:
                        progress_bar = st.progress(1 - tumor_probability)
                        st.write(f"Healthy scan confidence: {(1-tumor_probability)*100:.1f}%")
                    
                    # معلومات إضافية
                    st.markdown("---")
                    st.markdown(f"**⏱️ Processing Time:** {processing_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    else:
        st.info(" Please upload an MRI image to get started")
        
else:
    st.error("""
    ❌ Failed to load the model. Possible solutions:
    1. Check if the model file is compatible with TensorFlow 2.13.0
    2. Try restarting the application
    3. Verify the model file integrity
    """)

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI-powered application classifies MRI scans to detect potential brain tumors.
    
    --How to use:
    1. Upload a clear MRI scan image
    2. Wait for the analysis
    3. Review the results
    
    --Important Notes:
    - This tool is for screening purposes only
    - Always consult with medical professionals
    - Results should be verified by qualified radiologists
    """)
    
    st.header("Technical Info")
    st.write(f"Model file: {MODEL_PATH}")
    st.write(f"Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    st.write(f"TensorFlow version: {tf.__version__}")

st.markdown("---")
st.markdown("*Built with using Streamlit and TensorFlow*")

