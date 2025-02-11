import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


@st.cache_resource
def load_cancer_model():
    model = load_model("dl_model.h5")  
    return model

model = load_cancer_model()

# Image preprocessing 
def preprocess_image(img):
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array

# Streamlit UI
st.title("Breast Cancer Detection")
st.write("Upload an image to check for breast cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    

    result = "Malignant (Cancer Detected)" if prediction[0][0] > 0.5 else "Benign (No Cancer Detected)"
    confidence = float(prediction[0][0]) * 100 if prediction[0][0] > 0.5 else (1 - float(prediction[0][0])) * 100

    st.subheader("Prediction Result")
    st.write(f"**{result}**")
    st.write(f"Confidence: {confidence:.2f}%")
