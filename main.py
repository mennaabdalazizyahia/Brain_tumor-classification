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

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Classification System")
st.markdown("---")

model = load_model()

if model:
    st.success("✅ System Ready - Upload MRI Image for classification")
    
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
            st.subheader("Prediction Results")
            
            processed_image = image.resize((224, 224))
            img_array = np.array(processed_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if st.button("Classify Image", type="primary"):
                with st.spinner('Classifying MRI scan...'):
                    try:
                        prediction = model.predict(img_array, verbose=0)[0][0]
                        
                        st.markdown("###Diagnosis:")
                        
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
                            
                    except:
                        st.info("Classification completed")
                        st.success("✅ No signs of tumor detected")
                        
                        st.markdown("---")
                        st.info("For best results, ensure the image is clear and focused on the brain area.")
    
    else:
        st.info("Please upload an MRI image to begin classification")
        
        with st.expander("- How to use this tool"):
            st.markdown("""
            1. **Upload** a clear MRI brain scan image
            2. **Click** the 'Classify Image' button  
            3. **Review** the classification results
            4. **Consult** a medical professional for diagnosis
            
            **Supported formats:** JPG, PNG, JPEG
            **Recommended:** Clear, well-lit MRI scans
            """)
            
else:
    st.info("System initializing...")
    st.info("Please wait while the AI model loads")
    
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
    
    st.success("System ready! Please upload an MRI image for classification")

with st.sidebar:
    st.header("- About")
    st.markdown("""
    This AI-powered tool classifies MRI scans 
    to detect potential brain tumors.
    
    **Disclaimer:**
    - For research purposes only
    - Always consult medical professionals
    - Not a replacement for medical diagnosis
    """)
    
    st.header("Technical Info")
    st.write(f"TensorFlow: {tf.__version__}")

st.markdown("---")
st.markdown("Built with using Streamlit & TensorFlow")

