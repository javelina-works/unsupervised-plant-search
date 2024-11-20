import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import logging
from logging_utils import setup_logger

import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg" depending on your system


# Initialize the logger
logger = setup_logger(name="my_project_logger", log_level=logging.DEBUG)

# Where is your image?
geotiff_path = "input\ESPG-4326-orthophoto.tif"  # Replace with your file path



# Load the Geotiff Image
def load_image(file_path):
    """
    Load a multispectral image using rasterio.
    
    Args:
    - file_path (str): Path to the image file.
    
    Returns:
    - image (numpy.ndarray): 3D array of the image bands.
    - band_names (list): Names of bands if available (or generic labels).
    """
    with rasterio.open(file_path) as src:
        image = src.read()  # Shape: (bands, height, width)
        band_names = src.descriptions or [f"Band {i+1}" for i in range(image.shape[0])]
        
        # Check if band_names contains None values, and replace them with default names
        if all(name is None for name in band_names):
            band_names = [f"Band {i+1}" for i in range(image.shape[0])]
    
    return image, band_names

# Compute the average intensities for each band
def compute_band_intensities(image):
    """
    Compute the average intensity for each band in the image.
    
    Args:
    - image (numpy.ndarray): 3D array of the image bands.
    
    Returns:
    - intensities (numpy.ndarray): Array of mean intensities per band.
    """
    intensities = [np.mean(image[band]) for band in range(image.shape[0])]
    return intensities

# Visualize the spectrum of average intensities across bands
def plot_spectrum(intensities, band_names=None):
    """
    Plot the spectrum of average intensities across bands.
    
    Args:
    - intensities (list or numpy.ndarray): Average intensities for each band.
    - band_names (list): Optional list of band labels.
    """
    if band_names is None:
        band_names = [f"Band {i+1}" for i in range(len(intensities))]

    # Set the Seaborn style for the plot
    sns.set_context("notebook")  # Set context (larger font sizes, etc.)
    sns.set_style("whitegrid")   # Apply the 'whitegrid' style
    
    plt.figure(figsize=(10, 6))
    # Removed palette and added color assignment manually
    sns.barplot(x=band_names, y=intensities, color="blue")
    plt.title("Spectral Intensity Distribution", fontsize=16)
    plt.xlabel("Bands", fontsize=14)
    plt.ylabel("Average Intensity", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the image
    image, band_names = load_image(geotiff_path)

    # Compute average intensities
    intensities = compute_band_intensities(image)

    logger.debug(f"Intensities: {intensities}")
    logger.debug(f"Band Names: {band_names}")

    # Plot the spectrum
    plot_spectrum(intensities, band_names)
