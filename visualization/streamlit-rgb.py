import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app layout
st.title("Interactive RGBA Histogram and Image Viewer")
st.sidebar.header("Upload an Image")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file:
    # Open the file using rasterio
    with rasterio.open(uploaded_file) as src:
        image = src.read()  # Read all bands
        metadata = src.meta  # Read metadata

    # Ensure the image has at least 4 bands (RGBA)
    if src.count < 4:
        st.error("The uploaded image must have at least 4 bands (R, G, B, A).")
    else:
        # Extract RGBA bands
        red = image[0].astype("float32")
        green = image[1].astype("float32")
        blue = image[2].astype("float32")
        alpha = image[3].astype("float32")

        # Sidebar Controls for Interaction
        st.sidebar.subheader("Histogram Filters")
        
        # Checkbox to hide alpha channel
        hide_alpha = st.sidebar.checkbox("Hide Alpha Channel", value=False)

        # Slider to filter RGB values based on alpha
        alpha_range = st.sidebar.slider("Filter Alpha Range", 0.0, float(alpha.max()), (0.0, float(alpha.max())))

        # Slider to filter all histogram values
        value_range = st.sidebar.slider("Filter RGB and Alpha Values", 0.0, float(image.max()), (0.0, float(image.max())))

        # Apply filters
        valid_alpha = (alpha >= alpha_range[0]) & (alpha <= alpha_range[1])
        valid_values = (red >= value_range[0]) & (red <= value_range[1]) & \
                       (green >= value_range[0]) & (green <= value_range[1]) & \
                       (blue >= value_range[0]) & (blue <= value_range[1])

        valid_mask = valid_alpha & valid_values

        # Filter channels
        red_filtered = red[valid_mask]
        green_filtered = green[valid_mask]
        blue_filtered = blue[valid_mask]
        alpha_filtered = alpha[valid_mask] if not hide_alpha else None

        # Display the histogram
        st.write("### Combined Histogram for RGBA Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(red_filtered, bins=50, color="red", alpha=0.6, label="Red")
        ax.hist(green_filtered, bins=50, color="green", alpha=0.6, label="Green")
        ax.hist(blue_filtered, bins=50, color="blue", alpha=0.6, label="Blue")
        if not hide_alpha:
            ax.hist(alpha_filtered, bins=50, color="gray", alpha=0.6, label="Alpha")
        ax.set_title("Histogram of RGBA Values")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        # Display the RGBA image
        st.write("### RGBA Image")
        red_norm = red / red.max()
        green_norm = green / green.max()
        blue_norm = blue / blue.max()
        alpha_norm = alpha / alpha.max()
        rgba_image = np.stack([red_norm, green_norm, blue_norm, alpha_norm], axis=-1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgba_image)
        ax.axis("off")
        st.pyplot(fig)

        # Display metadata
        st.write("### Image Metadata")
        st.write(metadata)
