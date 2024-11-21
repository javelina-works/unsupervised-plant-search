import streamlit as st
import rasterio
import numpy as np
from matplotlib import pyplot as plt

# Function to read the GeoTIFF and convert it to RGB for display
def read_geotiff(uploaded_file):
    with rasterio.open(uploaded_file) as src:
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        
        # Stack bands into an RGB image
        rgb_image = np.dstack((red, green, blue))

        # Normalize the image to 8-bit (0-255) for display
        rgb_image = (rgb_image / np.max(rgb_image) * 255).astype(np.uint8)

        return rgb_image

# Streamlit app layout
st.title("Interactive GeoTIFF Viewer")
st.write("Upload a GeoTIFF file to interactively zoom and pan the image.")

# File upload widget
uploaded_file = st.file_uploader("Choose a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file is not None:
    # Read the uploaded GeoTIFF
    img_array = read_geotiff(uploaded_file)

    # Display the image interactively using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    ax.set_title("Interactive GeoTIFF Viewer")
    ax.axis("off")  # Hide axis for a cleaner display

    # Add zoom and pan tools to the figure
    st.pyplot(fig)
    st.write("Use the Matplotlib toolbar to zoom and pan directly on the image.")
