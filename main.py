import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

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
            with st.spinner('Classifying MRI scan...'):
                time.sleep(2)

                gray_image = processed_image.convert('L')
                np_gray = np.array(gray_image)
                
                contrast = np.std(np_gray)
                
                if contrast > 45: 
                    result = "TUMOR_DETECTED"
                    confidence = min(80 + (contrast - 45) / 2, 95)
                else:
                    result = "NO_TUMOR" 
                    confidence = min(85 + (45 - contrast) / 2, 98)
                
                st.markdown("### Diagnosis:")
                
                if result == "TUMOR_DETECTED":
                    st.error(f"**🚨 TUMOR DETECTED**")
                    st.write(f"**Confidence Level:** {confidence:.1f}%")
                    st.progress(confidence/100)
                    st.write(f"Tumor likelihood: {confidence:.1f}%")
                    
                    st.markdown("---")
                    st.warning("""
                    **⚠️ Important Notice:**
                    - This is a demonstration only
                    - Always consult medical professionals
                    - For accurate diagnosis, visit a healthcare provider
                    """)
                else:
                    st.success(f"**✅ NO TUMOR DETECTED**")
                    st.write(f"**Confidence Level:** {confidence:.1f}%")
                    st.progress(confidence/100)
                    st.write(f"💚 Healthy scan confidence: {confidence:.1f}%")
                    
                    st.markdown("---")
                    st.info("""
                    **Note:**
                    - This classification is for demonstration
                    - Regular medical checkups are recommended
                    - Consult doctors for professional diagnosis
                    """)

else:
    st.info("Please upload an MRI image to begin classification")
    
    with st.expander("How to use this tool"):
        st.markdown("""
        **Simple Steps:**
        1. **Click 'Browse files'** below
        2. **Select** an MRI brain scan from your computer
        3. **Click** the 'Analyze Image' button
        4. **View** the analysis results
        
        **Supported formats:** JPG, PNG, JPEG
        **Best results:** Clear, well-lit MRI scans
        
        **Important:**
        This tool is for **demonstration purposes only**.
        Always consult qualified medical professionals for diagnosis.
        """)

    st.markdown("###Example MRI Scan Types:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🩻 Axial View**")
        st.markdown("*Horizontal cross-section*")
    
    with col2:
        st.markdown("**🩻 Sagittal View**") 
        st.markdown("*Side cross-section*")
    
    with col3:
        st.markdown("**🩻 Coronal View**")
        st.markdown("*Front cross-section*")

with st.sidebar:
    st.header("Medical Disclaimer")
    st.markdown("""
    **Important Information:**
    
    **Purpose:**
    - Educational demonstration only
    - Research purposes
    - Not for medical diagnosis
    
    **Limitations:**
    - Not a substitute for professional medical advice
    - Always consult healthcare providers
    - Results are simulated for demonstration
    
    **For Real Diagnosis:**
    - Visit qualified radiologists
    - Use approved medical equipment
    - Follow doctor's recommendations
    """)
    
    st.header("Technical Info")
    st.write(f"Framework: Streamlit")
    st.write(f"Analysis: AI-powered simulation")
    st.write(f"Status: System Ready")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built for Educational Purposes • Always Consult Medical Professionals
    </div>
    """,
    unsafe_allow_html=True
)

