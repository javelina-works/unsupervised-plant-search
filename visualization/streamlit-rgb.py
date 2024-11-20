import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app layout
st.title("Interactive RGBA Line Chart and Image Viewer")
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

        # Determine max value for sliders and chart axes
        max_value = int(image.max())
        min_value = 0

        # Sidebar Controls for Interaction
        # Collapsible section for Alpha controls
        with st.sidebar.expander("Alpha Controls", expanded=False):
            # Checkbox to hide alpha channel (default True)
            hide_alpha = st.checkbox("Hide Alpha Channel", value=True)

            # Slider to filter RGB values based on alpha
            alpha_range = st.slider(
                "Filter Alpha Range",
                min_value=min_value,
                max_value=max_value,
                value=(min_value + 2, max_value),
                step=1,
            )

        # Slider to highlight value range with vertical bars
        highlight_range = st.sidebar.slider(
            "Highlight RGB Value Range",
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
            step=1,
        )

        # Apply filters
        valid_alpha = (alpha >= alpha_range[0]) & (alpha <= alpha_range[1])

        # Calculate the histogram for each channel
        def calculate_histogram(channel):
            # Flatten the channel and calculate the histogram
            counts, bin_edges = np.histogram(channel[valid_alpha], bins=np.arange(min_value, max_value + 2))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get bin centers (pixel values)
            return bin_centers, counts

        # Get histograms for each channel
        red_x, red_y = calculate_histogram(red)
        green_x, green_y = calculate_histogram(green)
        blue_x, blue_y = calculate_histogram(blue)
        if not hide_alpha:
            alpha_x, alpha_y = calculate_histogram(alpha)
        else:
            alpha_x, alpha_y = None, None

        # Display the line chart
        st.write("### RGBA Line Chart with Highlighted Range")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Fill area under the lines for each channel
        ax.fill_between(red_x, 0, red_y, color="red", alpha=0.3)
        ax.fill_between(green_x, 0, green_y, color="green", alpha=0.3)
        ax.fill_between(blue_x, 0, blue_y, color="blue", alpha=0.3)
        if not hide_alpha:
            ax.fill_between(alpha_x, 0, alpha_y, color="gray", alpha=0.3)

        # Plot line charts for each channel
        ax.plot(red_x, red_y, color="red", label="Red", alpha=0.6)
        ax.plot(green_x, green_y, color="green", label="Green", alpha=0.6)
        ax.plot(blue_x, blue_y, color="blue", label="Blue", alpha=0.6)
        if not hide_alpha:
            ax.plot(alpha_x, alpha_y, color="gray", label="Alpha", alpha=0.6)

        # Highlight regions outside the selected range
        ax.axvspan(min_value, highlight_range[0], color="gray", alpha=0.2, label="Grayed Out")
        ax.axvspan(highlight_range[1], max_value, color="gray", alpha=0.2)

        # Add vertical bars for the highlighted range
        ax.axvline(x=highlight_range[0], color="gray", linestyle="--", linewidth=2, label="Min Highlight")
        ax.axvline(x=highlight_range[1], color="gray", linestyle="--", linewidth=2, label="Max Highlight")

        # Customize the chart
        ax.set_title("Line Chart of RGBA Pixel Value Frequencies")
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
