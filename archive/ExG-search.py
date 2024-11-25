import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to the GeoTiff file
tiff_file = "input/ESPG-4326-orthophoto.tif"  # Replace with your file path

# Load the GeoTiff file and read bands
with rasterio.open(tiff_file) as src:
    # Read the first three bands (R, G, B)
    red_band = src.read(1).astype(np.float32)
    green_band = src.read(2).astype(np.float32)
    blue_band = src.read(3).astype(np.float32)

# Compute the Excess Green (ExG) index
exg_index = 2 * green_band - red_band - blue_band

# Normalize the ExG index for visualization
exg_normalized = (exg_index - np.min(exg_index)) / (np.max(exg_index) - np.min(exg_index))

# Plot the ExG index
plt.figure(figsize=(10, 6))
plt.imshow(exg_normalized, cmap='Greens')
plt.title('Excess Green (ExG) Index')
plt.colorbar(label='Normalized ExG Value')
plt.axis('off')
plt.show()

# Create a binary vegetation mask using a threshold
threshold = 0.2  # Adjust based on your image characteristics
vegetation_mask = exg_index > (threshold * np.max(exg_index))

# Plot the vegetation mask
plt.figure(figsize=(10, 6))
plt.imshow(vegetation_mask, cmap='gray')
plt.title('Vegetation Mask')
plt.axis('off')
plt.show()

# Optional: Save the vegetation mask as a new GeoTiff
output_path = 'vegetation_mask.tif'
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=vegetation_mask.shape[0],
    width=vegetation_mask.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=src.crs,
    transform=src.transform
) as dst:
    dst.write(vegetation_mask.astype(rasterio.uint8), 1)

print(f"Vegetation mask saved to {output_path}")
