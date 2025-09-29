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
        with st.spinner('Downloading model...'):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e1:
        st.warning(f"First method failed: {e1}")
        try:
            custom_objects = {
                'InputLayer': tf.keras.layers.InputLayer,
                'DTypePolicy': tf.keras.mixed_precision.Policy,
            }
            model = tf.keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                custom_objects=custom_objects
            )
            return model
        except Exception as e2:
            st.error(f"All methods failed: {e2}")
            return None

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Classification System")
st.markdown("---")

st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose an MRI scan image", 
    type=["jpg", "png", "jpeg", "JPG"],
    help="Select a clear MRI brain scan image"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="MRI Brain Scan", use_container_width=True)
        st.info(f"**Image Details:** {image.size[0]}x{image.size[1]} pixels")
    
    with col2:
        st.subheader("Classification Results")
        
        processed_image = image.resize((224, 224))
        img_array = np.array(processed_image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Classify Image", type="primary"):
            model = load_model()
            
            if model is not None:
                with st.spinner('Analyzing MRI scan...'):
                    try:
                        prediction = model.predict(img_array, verbose=0)[0][0]
                        
                        st.markdown("### Diagnosis:")
                        
                        if prediction > 0.5:
                            st.error(f"**🚨 TUMOR DETECTED**")
                            st.write(f"**Confidence Level:** {prediction*100:.2f}%")
                            st.progress(prediction)
                            st.write(f"Tumor likelihood: {prediction*100:.1f}%")
                        else:
                            st.success(f"**✅ NO TUMOR DETECTED**")
                            st.write(f"**Confidence Level:** {(1-prediction)*100:.2f}%")
                            st.progress(1 - prediction)
                            st.write(f"Healthy scan confidence: {(1-prediction)*100:.1f}%")
                            
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.info("Using demo analysis...")
                        demo_result = np.random.choice([True, False])
                        if demo_result:
                            st.error("**🚨 TUMOR DETECTED** (Demo)")
                            st.write("**Note:** This is a demo result")
                        else:
                            st.success("**✅ NO TUMOR DETECTED** (Demo)")
                            st.write("**Note:** This is a demo result")
            else:
                st.warning("Model not available - Using demo mode")
                demo_result = np.random.choice([True, False])
                if demo_result:
                    st.error("**🚨 TUMOR DETECTED** (Demo)")
                else:
                    st.success("**✅ NO TUMOR DETECTED** (Demo)")
                st.info("Please check model compatibility")

else:
    st.info("Please upload an MRI image to begin classification")
    
    with st.expander("How to use this tool"):
        st.markdown("""
        **Steps:**
        1. **Click 'Browse files'** below
        2. **Select** an MRI brain scan image from your computer
        3. **Click** the 'Classify Image' button
        4. **View** the analysis results
        
        **Supported formats:** JPG, PNG, JPEG
        **File size limit:** 200MB
        """)

with st.sidebar:
    st.header("System Status")
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.write(f"Model size: {file_size:.2f} MB")
        st.write(f"Model file: Present")
    else:
        st.write("Model file: Not found")
    
    st.write(f"TensorFlow: {tf.__version__}")
    
    st.header("About")
    st.markdown("""
    AI-powered brain tumor detection
    from MRI scans.
    
    **For research purposes only**
    """)

st.markdown("---")
st.markdown("Built with Streamlit & TensorFlow")
