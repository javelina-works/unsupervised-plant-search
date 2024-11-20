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



# Load the Geotiff Image (with alpha channel removal)
def load_image(file_path):
    """
    Load a multispectral image using rasterio, and remove the alpha channel if present.
    
    Args:
    - file_path (str): Path to the image file.
    
    Returns:
    - image (numpy.ndarray): 3D array of the image bands (RGB).
    - band_names (list): Names of bands if available (or generic labels).
    - alpha_channel (numpy.ndarray): 2D array of alpha channel (transparency values).
    """
    with rasterio.open(file_path) as src:
        image = src.read()  # Shape: (bands, height, width)
        band_names = src.descriptions or [f"Band {i+1}" for i in range(image.shape[0])]
        
        # Check if image has an alpha channel (4th band) and remove it
        if image.shape[0] == 4:
            # The first 3 bands are RGB and the 4th is alpha (transparency)
            rgb_image = image[:3, :, :]  # Only keep the first 3 bands (RGB)
            alpha_channel = image[3, :, :]  # Extract the alpha channel
            band_names = band_names[:3]  # Keep only the first 3 band names (RGB)
        else:
            # If no alpha channel exists, just return the RGB image
            rgb_image = image[:3, :, :]
            alpha_channel = None

        # Check if band_names contains None values, and replace them with default names
        if all(name is None for name in band_names):
            band_names = [f"Band {i+1}" for i in range(image.shape[0])]

    return rgb_image, alpha_channel, band_names

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

# Visualize RGB Channels as Continuous Spectrum
def visualize_rgb_channels(image):
    """
    Visualize the RGB channels of the image as a continuous spectrum.
    
    Args:
    - image (numpy.ndarray): 3D array of the image bands (RGB).
    """
    if image.shape[0] < 3:
        print("Image does not have RGB channels.")
        return

    # Extract the RGB channels (assuming the first three bands correspond to R, G, B)
    red_channel = image[0, :, :]
    green_channel = image[1, :, :]
    blue_channel = image[2, :, :]

    # Plot the RGB channels as images with continuous color gradients
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Red channel
    ax[0].imshow(red_channel, cmap='Reds', interpolation='nearest')
    ax[0].set_title('Red Channel', fontsize=14)
    ax[0].axis('off')  # Turn off axis labels

    # Green channel
    ax[1].imshow(green_channel, cmap='Greens', interpolation='nearest')
    ax[1].set_title('Green Channel', fontsize=14)
    ax[1].axis('off')

    # Blue channel
    ax[2].imshow(blue_channel, cmap='Blues', interpolation='nearest')
    ax[2].set_title('Blue Channel', fontsize=14)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


# Compute and plot histograms for RGB intensities (after applying alpha mask)
def plot_rgb_histograms(image, alpha_channel):
    """
    Plot the histograms of the RGB intensity distributions for each pixel,
    ignoring pixels with an alpha value of 1 (fully transparent), and fill the area below each curve.
    
    Args:
    - image (numpy.ndarray): 3D array of the image bands (RGB).
    - alpha_channel (numpy.ndarray): 2D array of the alpha channel (transparency values).
    """
    if image.shape[0] < 3:
        print("Image does not have RGB channels.")
        return

    # Apply mask to ignore RGB values for fully transparent pixels (alpha == 1)
    if alpha_channel is not None:
        # Create a mask where alpha is not 1 (i.e., non-transparent pixels)
        mask = alpha_channel != 0
        # Apply the mask to the RGB channels
        red_channel = image[0, :, :][mask].flatten()
        green_channel = image[1, :, :][mask].flatten()
        blue_channel = image[2, :, :][mask].flatten()
    else:
        # If no alpha channel exists, we can proceed with the whole image
        red_channel = image[0, :, :].flatten()
        green_channel = image[1, :, :].flatten()
        blue_channel = image[2, :, :].flatten()

    # Create a single plot with three histograms
    plt.figure(figsize=(10, 6))

    # Red Channel Histogram
    plt.hist(red_channel, bins=256, color='red', alpha=0.7, range=(0, 255), label='Red Channel')

    # Green Channel Histogram
    plt.hist(green_channel, bins=256, color='green', alpha=0.7, range=(0, 255), label='Green Channel')

    # Blue Channel Histogram
    plt.hist(blue_channel, bins=256, color='blue', alpha=0.7, range=(0, 255), label='Blue Channel')

    # Customize the plot
    plt.title('RGB Histograms with Area Filled')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()





if __name__ == "__main__":
    # Load the image (with alpha channel removal)
    image, alpha_channel, band_names = load_image(geotiff_path)

    # Compute average intensities
    intensities = compute_band_intensities(image)

    logger.debug(f"Intensities: {intensities}")
    logger.debug(f"Band Names: {band_names}")

    # Plot the spectrum
    # plot_spectrum(intensities, band_names)

    # Visualize the RGB channels
    # visualize_rgb_channels(image)

    # Plot the RGB histograms (ignoring alpha transparency)
    plot_rgb_histograms(image, alpha_channel)
