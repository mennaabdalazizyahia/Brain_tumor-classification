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
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {e}")
else:
    st.success("Model file already exists!")

if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024) 
    st.info(f"Model file size: {file_size:.2f} MB")
    
    if file_size < 1:
        st.warning("Model file seems too small. There might be an issue with the download.")

