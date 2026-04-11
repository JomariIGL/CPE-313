import streamlit as st
import tensorflow as tf
import os
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()  # ✅ assign the returned model here

st.write("# Banana Artificial vs Natural Classification")
file = st.file_uploader("Choose banana photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Artificial', 'Natural']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)