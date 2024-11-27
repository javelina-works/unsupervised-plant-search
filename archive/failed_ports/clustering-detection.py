import streamlit as st
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Function to load GeoTIFF and extract RGB bands
def load_geotiff(file):
    with rasterio.open(file) as src:
        red = src.read(1).astype(float)
        green = src.read(2).astype(float)
        blue = src.read(3).astype(float)
    return red, green, blue

# Function to calculate Excess Green Index (ExG)
def calculate_exg(red, green, blue):
    exg = 2 * green - red - blue
    return exg

# Function to normalize image
def normalize(image):
    normalized = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized.astype(np.uint8)

# Function to threshold vegetation mask
def threshold_vegetation(exg, threshold_value):
    _, binary_mask = cv2.threshold(exg, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_mask

# Function to refine mask
def refine_mask(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

# Streamlit App
st.title("Interactive Vegetation Analysis Tool")

# Upload GeoTIFF File
uploaded_file = st.file_uploader("Upload a GeoTIFF file (RGB bands required)", type=["tif", "tiff"])

if uploaded_file:
    # Load image
    red, green, blue = load_geotiff(uploaded_file)
    
    # Display original image
    st.subheader("Original RGB Image")
    rgb_image = np.dstack((normalize(red), normalize(green), normalize(blue))).astype(np.uint8)
    st.image(rgb_image, caption="RGB Image", use_column_width=True)

    # Calculate ExG
    exg = calculate_exg(red, green, blue)
    normalized_exg = normalize(exg)

    # Interactive threshold slider
    st.subheader("Threshold Vegetation Mask")
    threshold_value = st.slider("Set Threshold Value", 0, 255, 127)
    vegetation_mask = threshold_vegetation(normalized_exg, threshold_value)
    st.image(vegetation_mask, caption="Binary Vegetation Mask", use_column_width=True, clamp=True)

    # Interactive kernel size slider
    st.subheader("Refine Mask")
    kernel_size = st.slider("Set Kernel Size for Morphological Operations", 1, 20, 5)
    refined_mask = refine_mask(vegetation_mask, kernel_size)
    st.image(refined_mask, caption="Refined Vegetation Mask", use_column_width=True, clamp=True)

    # Download refined mask
    st.subheader("Download Refined Mask")
    mask_bytes = BytesIO()
    mask_image = cv2.imencode('.png', refined_mask)[1].tobytes()
    st.download_button(
        label="Download Mask",
        data=mask_image,
        file_name="vegetation_mask.png",
        mime="image/png"
    )
