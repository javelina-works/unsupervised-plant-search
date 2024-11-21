import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate NDVI
def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)

# Function to calculate other vegetation indices (e.g., EVI, GCI)
def calculate_evi(nir, red, blue):
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

def calculate_gci(nir, green):
    return nir / green - 1

# Streamlit app layout
st.title("Vegetation Index Visualizer and Histogram Viewer")
st.sidebar.header("Upload Georeferenced Image")

uploaded_file = st.sidebar.file_uploader("Upload a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file:
    with rasterio.open(uploaded_file) as src:
        # Read the image bands
        image = src.read()
        bands = src.count
        metadata = src.meta

    st.write(f"Image loaded with {bands} bands.")
    st.write(f"Metadata: {metadata}")

    # Select indices to calculate
    index = st.sidebar.selectbox(
        "Choose a vegetation index",
        ("NDVI", "EVI", "GCI")
    )

    # Get band assignment
    nir_band = st.sidebar.number_input("NIR Band", min_value=1, max_value=bands, value=4)
    red_band = st.sidebar.number_input("Red Band", min_value=1, max_value=bands, value=3)
    green_band = st.sidebar.number_input("Green Band", min_value=1, max_value=bands, value=2)
    blue_band = st.sidebar.number_input("Blue Band", min_value=1, max_value=bands, value=1)

    # Extract the necessary bands
    nir = image[nir_band - 1].astype("float32")
    red = image[red_band - 1].astype("float32")
    green = image[green_band - 1].astype("float32")
    blue = image[blue_band - 1].astype("float32")

    # Calculate selected index
    if index == "NDVI":
        result = calculate_ndvi(nir, red)
    elif index == "EVI":
        result = calculate_evi(nir, red, blue)
    elif index == "GCI":
        result = calculate_gci(nir, green)

    # Display vegetation index
    st.subheader(f"{index} Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(result, cmap="RdYlGn")
    ax.set_title(index)
    fig.colorbar(cax, ax=ax, orientation="vertical")
    st.pyplot(fig)

    # Add histogram viewer
    st.subheader("RGB Channel Histograms")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    axs[0].hist(red.flatten(), bins=50, color='red', alpha=0.7)
    axs[0].set_title("Red Channel")
    axs[0].set_xlabel("Pixel Value")
    axs[0].set_ylabel("Frequency")
    
    axs[1].hist(green.flatten(), bins=50, color='green', alpha=0.7)
    axs[1].set_title("Green Channel")
    axs[1].set_xlabel("Pixel Value")
    
    axs[2].hist(blue.flatten(), bins=50, color='blue', alpha=0.7)
    axs[2].set_title("Blue Channel")
    axs[2].set_xlabel("Pixel Value")
    
    plt.tight_layout()
    st.pyplot(fig)
