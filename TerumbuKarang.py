# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:42:43 2023

@author: USER
"""

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

st.title("Coral Reefs Classification")

st.markdown(
    """
    <p style="text-align:justify;">
    Corals are marine invertebrates belonging to the phylum Cnidaria and the class Anthozoa. Renowned for their vibrant colors and diverse patterns, they play a crucial role in the marine ecosystem. Forming colonies that serve as habitats for a wide array of marine species, corals also contribute to the protection of coastlines from storms and erosion. However, corals are currently confronted with substantial threats stemming from climate change, pollution, and overfishing, leading to coral bleaching and mortality. Coral reefs, constitute a vast ecosystem, with estimates suggesting that their expanse accounts for less than 1% of the total area of the oceans. This ecosystem serves as the home for approximately 25% of marine species. Coral reefs play a crucial role in sequestering carbon in seawater and act as natural barriers, providing protection to coastal areas.
    """,
    unsafe_allow_html=True
)

# Use columns to organize the layout
col1, col2, col3 = st.columns(3)

# Load and display images in separate columns
with col1:
    img1 = Image.open('2.png')
    st.image(img1, caption='Dead', use_column_width=True)

with col2:
    img2 = Image.open('21.png')
    st.image(img2, caption='Bleached', use_column_width=True)

with col3:
    img3 = Image.open('15.png')
    st.image(img3, caption='healthy', use_column_width=True)
    
st.markdown(
    """
    <p style="text-align:justify;">
    The classification process will use the Transfer Learning CNN method, namely the DenseNet architecture. DenseNet was chosen because of its strong inter-layer feature relationships, enhancing the learning of complex features and optimizing the use of parameters in the architecture.
    """,
    unsafe_allow_html=True
)

img4 = Image.open('DenseNet.png')
st.image(img4, caption='Dead', use_column_width=True)

st.markdown(
    """
    <p style="text-align:justify;">
    The dataset used for building model classification is consists of a total of 1582 coral reef image data, further divided into 3 classes: healthy, bleached, and dead. The distribution of data per class includes 712 healthy data, 720 bleached data, and 150 dead data
    """,
    unsafe_allow_html=True
)

st.title("Classification Process")
# Load the trained model
model = load_model("DenseNet_TerumbuKarang1.h5")

# File uploader for user input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for prediction
    img_size = 224  # Adjust the image size based on DenseNet's input requirements
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    # Display the predicted class
    class_labels = ["Bleached", "Dead", "Healthy"]
    st.success(f"Predicted Class: {class_labels[predicted_class]}")