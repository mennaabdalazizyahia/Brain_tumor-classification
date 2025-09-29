import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import warnings
warnings.filterwarnings('ignore')

FILE_ID = "1MimIt5qq_NyzxqGoZIBakqv4JhdpkjRL"
MODEL_PATH = "brain_tumor_model.h5"

def create_compatible_model():
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model... (This may take a few minutes)'):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.info(f"Model file size: {file_size:.2f} MB")
    
    with st.spinner('Attempting to load model...'):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            st.success("Model loaded successfully!")
            return model
        except Exception as e1:
            st.warning(f"Standard loading failed: {str(e1)[:100]}...")
            
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
                st.success("Model loaded with custom objects!")
                return model
            except Exception as e2:
                st.warning(f"Custom objects failed: {str(e2)[:100]}...")
                
                try:
                    import tensorflow.compat.v1 as tf_compat
                    tf_compat.disable_v2_behavior()
                    
                    with tf_compat.Session() as sess:
                        model = tf_compat.keras.models.load_model(MODEL_PATH)
                    st.success("Model loaded with TF v1 compatibility!")
                    return model
                except Exception as e3:
                    st.error(f"All loading methods failed")
                    st.info("🔄 Creating a compatible model for demonstration...")
                    
                    demo_model = create_compatible_model()
                    if demo_model:
                        st.warning("Using demo model - Upload functionality available but with random predictions")
                        return demo_model
                    return None

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection System")
st.markdown("---")
st.write(f"**TensorFlow Version:** {tf.__version__}")

model = load_model()

if model is not None:
    is_demo_model = len(model.layers) < 6  
    
    if is_demo_model:
        st.warning("""
        **Demo Mode Active**
        - The original model is not compatible
        - Using demonstration model
        - Upload and interface work normally
        - Predictions are for display only
        """)
    else:
        st.success("Original model loaded successfully! Ready for predictions.")
    
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
            
            st.info(f"**Image Info:** {image.size} pixels | {image.format} format")
        
        with col2:
            st.subheader("Classification Results")
            
            processed_image = image.resize((224, 224))
            img_array = np.array(processed_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner('Analyzing MRI scan...'):
                try:
                    start_time = time.time()
                    
                    if is_demo_model:
                        tumor_probability = np.random.uniform(0.1, 0.9)
                        processing_time = time.time() - start_time
                        st.warning("**Demo Prediction** - Not actual medical analysis")
                    else:
                        prediction = model.predict(img_array, verbose=0)
                        processing_time = time.time() - start_time
                        tumor_probability = float(prediction[0][0])
                    
                    st.markdown("###Diagnosis Results:")
                    
                    if tumor_probability > 0.5:
                        st.error(f"**🚨 TUMOR DETECTED**")
                        st.write(f"**Confidence Level:** {tumor_probability*100:.2f}%")
                    else:
                        st.success(f"**✅ NO TUMOR DETECTED**")
                        st.write(f"**Confidence Level:** {(1-tumor_probability)*100:.2f}%")
                    
                    st.markdown("###Confidence Meter:")
                    progress_value = tumor_probability if tumor_probability > 0.5 else (1 - tumor_probability)
                    st.progress(float(progress_value))
                    
                    if tumor_probability > 0.5:
                        st.write(f"Tumor likelihood: {tumor_probability*100:.1f}%")
                    else:
                        st.write(f"Healthy scan confidence: {(1-tumor_probability)*100:.1f}%")
                    
                    st.markdown("---")
                    st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                    
                    if is_demo_model:
                        st.info("""
                        **For real predictions:**
                        - Install keras-core: `pip install keras-core`
                        - Or use Google Colab with latest TF
                        - Contact model provider for compatible version
                        """)
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    else:
        st.info("Please upload an MRI image to get started")
        
else:
    st.error("""
    **❌ Final Solution Required:**
    
    **Option 1: Install Keras 3**
    ```bash
    pip install keras-core
    ```
    
    **Option 2: Use Google Colab**
    ```python
    !pip install tensorflow==2.15.0 keras-core
    ```
    
    **Option 3: Request compatible model format from provider**
    """)

# معلومات المساعدة
with st.sidebar:
    st.header("Troubleshooting")
    st.markdown("""
    **Current Issue:**
    - Model created with Keras 3
    - Incompatible with TF 2.15.0
    - Requires `keras-core`
    
    **Quick Fix:**
    ```bash
    pip install keras-core
    restart app
    ```
    """)
    
    st.header("Technical Info")
    st.write(f"TF Version: {tf.__version__}")
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.write(f"Model Size: {size:.2f} MB")

st.markdown("---")
st.markdown("**Note:** For medical use, ensure model compatibility and validation")
