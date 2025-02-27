import os, sys
import streamlit as st
from PIL import Image

#Setup path
path_this = os.path.abspath(os.path.dirname(__file__))
path_root = os.path.abspath(os.path.join(path_this, '..'))
sys.path.insert(0, path_root)

# load config
from config import settings
from utils import sample_image

# load SVC and CNN inference module
from inference.SVCPredict import SVCPredict
from inference.CNNPredict import CNNPredict

# setup logo and page title
st.logo("assets/logo.png", icon_image="assets/logo.png", size="large")
st.title("AI Model Inference App")

# inference model selection between SVC and CNN
model_type = st.selectbox("Select Model", ["SVC", "CNN"])

if model_type=="SVC":
    model = SVCPredict()
elif model_type=="CNN":
    model = CNNPredict()

# Image Input Methode
option = st.selectbox("Select Image Input", ["Sample Image", "Upload"])

if option=="Sample Image":
    image = st.selectbox("Select Image", sample_image.images_path())
    
elif option=="Upload":
    image = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if st.button("Predict"):
    with st.spinner():
        image = Image.open(image)
        predict_result = model.predict_image(image_pil=image)

    col1, col2 = st.columns(2, gap="large", vertical_alignment="top", border=True)    
    col1.header("Image", divider="rainbow")
    col1.image(image, use_container_width=True)
    col2.header("Prediction Result", divider="rainbow")
    col2.json(predict_result)