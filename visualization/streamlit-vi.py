import streamlit as st
import rasterio
import numpy as np
from matplotlib.colors import Normalize
import folium
from streamlit_folium import st_folium

def calculate_vegetation_index(image, index_type):
    """
    Calculate vegetation indices using RGB bands.
    """
    red = image[0].astype("float32")
    green = image[1].astype("float32")
    blue = image[2].astype("float32")
    epsilon = 1e-6  # Small value to prevent division by zero

    if index_type == "Synthetic NDVI":
        index = (green - red) / (green + red + epsilon)
    elif index_type == "Green Normalized Difference Index (GNDVI)":
        index = (green - blue) / (green + blue + epsilon)
    elif index_type == "Red Green Ratio Index (RGRI)":
        index = red / (green + epsilon)
    else:
        index = np.zeros_like(red)  # Default to zeros for unsupported indices

    return np.nan_to_num(index, nan=0.0)

# Streamlit app layout
st.title("Interactive Vegetation Indices Viewer")

# Upload Section
st.sidebar.header("Upload GeoTIFF")
uploaded_file = st.sidebar.file_uploader("Upload a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file:
    with rasterio.open(uploaded_file) as src:
        image = src.read()
        metadata = src.meta

    # Section 1: Vegetation Index Selection
    st.subheader("Vegetation Index Controls")
    index_type = st.selectbox(
        "Select Vegetation Index", 
        ["Synthetic NDVI", "Green Normalized Difference Index (GNDVI)", "Red Green Ratio Index (RGRI)"]
    )
    index_range = st.slider(
        "Adjust Index Range",
        min_value=-1.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.1,
    )

    # Calculate vegetation index
    vegetation_index = calculate_vegetation_index(image, index_type)

    # Apply range filter
    filtered_index = np.clip(vegetation_index, index_range[0], index_range[1])

    # Section 2: Interactive Map
    st.subheader("Interactive Vegetation Index Map")

    # Normalize the index to [0, 255] for visualization
    norm = Normalize(vmin=index_range[0], vmax=index_range[1])
    normalized_index = (norm(filtered_index) * 255).astype(np.uint8)

    # Prepare RGB image for Folium overlay
    index_image = np.zeros((filtered_index.shape[0], filtered_index.shape[1], 3), dtype=np.uint8)
    index_image[:, :, 0] = normalized_index  # Red channel
    index_image[:, :, 1] = normalized_index  # Green channel
    index_image[:, :, 2] = normalized_index  # Blue channel

    # Set up the Folium map
    center_lat = (src.bounds.top + src.bounds.bottom) / 2
    center_lon = (src.bounds.left + src.bounds.right) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add overlay to the map
    folium.raster_layers.ImageOverlay(
        image=index_image,
        bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]],
        opacity=0.7,
    ).add_to(m)

    # Display the map in Streamlit
    st_data = st_folium(m, width=700, height=500)

    # Metadata at the bottom
    st.write("### Image Metadata")
    st.json(metadata)
