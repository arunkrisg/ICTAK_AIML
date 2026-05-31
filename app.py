
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# light background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# load model and class names
model = tf.keras.models.load_model("final_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

# simple page title
st.title("Scenary Classification App")
st.write("Upload an image to predict the scene type")

# upload image
uploaded_file = st.file_uploader(
    "Choose an image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    # preprocess image
    img = image.resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # predict
    pred       = model.predict(arr)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0]) * 100

    # show result
    st.write("Uploaded Image is :", pred_class)
    st.write("Model Confidence  :", round(confidence, 2), "%")
