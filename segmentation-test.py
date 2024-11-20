import rasterio
import numpy as np
import matplotlib.pyplot as plt
import logging
from logging_utils import setup_logger


# Where is your image?
geotiff_path = "input\ESPG-4326-orthophoto.tif"  # Replace with your file path

# How much LOG??
LOG_LEVEL=logging.INFO



# Load the Geotiff
def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        logger.debug(f"Number of image bands: {src.count}")
        logger.debug(f"Band descriptions: {src.descriptions}")

        orthophoto = src.read()  # Read all bands
        profile = src.profile   # Save geotiff profile (metadata)
    return orthophoto, profile

# Compute NDVI
def compute_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)  # Prevent division by zero
    return ndvi


# Visualize Bands
def visualize_bands(orthophoto):
    num_bands = orthophoto.shape[0]
    plt.figure(figsize=(15, 5))
    for i in range(num_bands):
        plt.subplot(1, num_bands, i + 1)
        plt.imshow(orthophoto[i], cmap="gray")
        plt.title(f"Band {i + 1}")
        plt.colorbar()
    plt.tight_layout()
    plt.show()







# =====================================
# Main Execution
# =====================================

if __name__ == "__main__":

    logger = setup_logger(name="segmentation_logger", log_level=LOG_LEVEL)
    
    # Step 1: Load Geotiff
    orthophoto, profile = load_geotiff(geotiff_path)

    # Step 2: Visualize bands to identify Red and NIR
    visualize_bands(orthophoto)

    # Step 3: Select bands manually (assuming 3 = Red, 4 = NIR)
    red_band = orthophoto[2]  # Red is usually band 3
    nir_band = orthophoto[3]  # NIR is usually band 4

    # Step 4: Compute NDVI
    ndvi = compute_ndvi(nir_band, red_band)

    # Step 5: Visualize NDVI
    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi, cmap="RdYlGn")
    plt.colorbar(label="NDVI")
    plt.title("NDVI Computation")
    plt.axis("off")
    plt.show()
