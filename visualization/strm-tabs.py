import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Streamlit app layout
st.title("Interactive Image Viewer: RGBA and Vegetation Indices")

# Left sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file:
    with rasterio.open(uploaded_file) as src:
        image = src.read()  # Read all bands
        metadata = src.meta

    # Create two columns for main layout: center and right "sidebar"
    col1, col2 = st.columns([3, 1], gap="medium")

    with col1:
        # Tabs for RGBA and Vegetation Indices
        tab1, tab2 = st.tabs(["RGBA Mode", "Vegetation Indices"])

        with tab1:
            st.header("RGBA Mode")
            if src.count < 4:
                st.error("RGBA mode requires at least 4 bands (R, G, B, A).")
            else:
                # Extract RGBA bands
                red = image[0].astype("float32")
                green = image[1].astype("float32")
                blue = image[2].astype("float32")
                alpha = image[3].astype("float32")

                # Create RGBA histogram
                st.write("### RGBA Histogram")
                fig, ax = plt.subplots(figsize=(10, 6))
                for channel, color, label in zip(
                    [red, green, blue, alpha],
                    ["red", "green", "blue", "gray"],
                    ["Red", "Green", "Blue", "Alpha"],
                ):
                    counts, bins = np.histogram(channel, bins=256, range=(0, 255))
                    ax.plot((bins[:-1] + bins[1:]) / 2, counts, label=label, color=color)
                ax.legend()
                st.pyplot(fig)

                # Display RGBA image
                st.write("### RGBA Image")
                rgba_image = np.stack([red, green, blue, alpha], axis=-1)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(rgba_image / 255)
                ax.axis("off")
                st.pyplot(fig)

        with tab2:
            st.header("Vegetation Indices")
            if src.count < 4:
                st.error("Vegetation Indices mode requires at least 4 bands (e.g., NIR and Red).")
            else:
                # Compute NDVI
                red = image[0].astype("float32")
                nir = image[3].astype("float32")
                ndvi = (nir - red) / (nir + red)
                ndvi = np.nan_to_num(ndvi, nan=0.0)

                # NDVI Histogram
                st.write("### NDVI Histogram")
                fig, ax = plt.subplots(figsize=(10, 6))
                counts, bins = np.histogram(ndvi, bins=256, range=(-1, 1))
                ax.fill_between((bins[:-1] + bins[1:]) / 2, counts, color="green", alpha=0.5)
                ax.set_title("NDVI Histogram")
                st.pyplot(fig)

                # NDVI Image
                st.write("### NDVI Image")
                cmap = plt.cm.viridis
                norm = Normalize(vmin=-1, vmax=1)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(cmap(norm(ndvi)))
                ax.axis("off")
                st.pyplot(fig)

    # Right "sidebar" for mode-specific settings
    with col2:
        st.header("Mode Settings")
        if tab1:
            st.subheader("RGBA Settings")
            hide_alpha = st.checkbox("Hide Alpha Channel", value=True)
            alpha_range = st.slider(
                "Alpha Range",
                min_value=0,
                max_value=255,
                value=(50, 200),
                step=1,
            )
        elif tab2:
            st.subheader("Vegetation Index Settings")
            ndvi_range = st.slider(
                "NDVI Range",
                min_value=-1.0,
                max_value=1.0,
                value=(-0.2, 0.8),
                step=0.1,
            )

    # Metadata at the bottom
    st.write("### Image Metadata")
    st.write(metadata)
