import rasterio
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

# Step 1: Load the GeoTIFF file
geotiff_path = "input\ESPG-4326-orthophoto.tif"

with rasterio.open(geotiff_path) as src:
    # Read the first band (modify as needed for multi-band data)
    band1 = src.read(1)
    # Normalize the data for better visualization
    band1_normalized = (band1 - np.min(band1)) / (np.max(band1) - np.min(band1))

# Step 2: Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(
    band1_normalized,
    cmap="viridis",
    interpolation="nearest"
)
cbar = plt.colorbar(im, ax=ax, orientation="vertical", label="Normalized Value")
ax.set_title("GeoTIFF Visualization")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Step 3: Render the plot interactively using mpld3
mpld3.show()
