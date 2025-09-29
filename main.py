import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import h5py
import json

FILE_ID = "1MimIt5qq_NyzxqGoZIBakqv4JhdpkjRL"
MODEL_PATH = "brain_tumor_model.h5"

def fix_model_config():
    try:
        with h5py.File(MODEL_PATH, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config = f.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                model_config = model_config.replace('"batch_shape":', '"batch_input_shape":')
                model_config = model_config.replace("'batch_shape':", "'batch_input_shape':")
                
                model_config = model_config.replace('"class_name": "InputLayer"', '"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3]}')
                
                f.attrs['model_config'] = model_config.encode('utf-8')
                return True
        return False
    except Exception as e:
        st.warning(f"Config fix attempt: {e}")
        return False

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model... (This may take a few minutes)'):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.info(f"Model file size: {file_size:.2f} MB")
    
    with st.spinner('Loading model...'):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"First attempt failed: {e}")
            
            try:
                st.info("Attempting to fix model configuration...")
                if fix_model_config():
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    st.success("Model loaded after configuration fix!")
                    return model
            except Exception as e2:
                st.error(f"Second attempt failed: {e2}")
                
            try:
                st.info("Trying with custom objects...")
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                }
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    compile=False,
                    custom_objects=custom_objects
                )
                st.success("Model loaded with custom objects!")
                return model
            except Exception as e3:
                st.error(f"Third attempt failed: {e3}")
                return None

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection System")
st.markdown("---")
st.write(f"**TensorFlow Version:** {tf.__version__}")
model = load_model()

if model is not None:
    st.success("Model loaded successfully! Ready for predictions.")
    
    st.subheader("Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image", 
        type=["jpg", "png", "jpeg", "JPG"],
        help="Upload a clear MRI scan image for classification"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Original MRI Scan", use_column_width=True)
            
            st.info(f"Image size: {image.size} | Format: {image.format}")
        
        with col2:
            st.subheader("Classification Results")
            
            processed_image = image.resize((224, 224))
            img_array = np.array(processed_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner('Classifying MRI scan...'):
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
                    
                    st.markdown("### Confidence Meter:")
                    if tumor_probability > 0.5:
                        progress_bar = st.progress(tumor_probability)
                        st.write(f"Tumor likelihood: {tumor_probability*100:.1f}%")
                    else:
                        progress_bar = st.progress(1 - tumor_probability)
                        st.write(f"Healthy scan confidence: {(1-tumor_probability)*100:.1f}%")
                    
                    st.markdown("---")
                    st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    else:
        st.info("Please upload an MRI image to get started")
        
else:
    st.error("""
    **❌ Failed to load the model. The model file might be:**
    
    - Created with a very old TensorFlow version
    - Corrupted during download
    - Incompatible with current TensorFlow
    
    **Please try re-downloading the model or contact the model provider.**
    """)

with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI-powered application analyzes MRI scans to detect potential brain tumors.
    
    **How to use:**
    1. Upload a clear MRI scan image
    2. Wait for the analysis
    3. Review the results
    """)
    
    st.header("Technical Info")
    st.write(f"TensorFlow: {tf.__version__}")
    st.write(f"Model: {MODEL_PATH}")

st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow")
