import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.predict import load_model, predict_image

st.set_page_config(page_title="Fashion-MNIST Classifier")
st.title("Fashion-MNIST Classifier")
st.write("Upload a clothing image and see the model's prediction.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    predicted_class, confidence = predict_image(model, image)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.1f}%")
    st.progress(confidence)