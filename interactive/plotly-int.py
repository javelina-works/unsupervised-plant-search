import rasterio
import numpy as np
import plotly.express as px

# Step 1: Load the GeoTIFF file
geotiff_path = "input\ESPG-4326-orthophoto.tif"  # Replace with your file path

with rasterio.open(geotiff_path) as src:
    # Read the first band (or modify to handle multiple bands)
    band1 = src.read(1)
    # Normalize the data for better visualization
    band1_normalized = (band1 - np.min(band1)) / (np.max(band1) - np.min(band1))

# Step 2: Create an interactive Plotly figure
fig = px.imshow(band1_normalized,
                color_continuous_scale="viridis",
                labels={"x": "Longitude", "y": "Latitude", "color": "Value"},
                title="GeoTIFF Visualization")

# Step 3: Show the plot
fig.update_layout(
    coloraxis_colorbar=dict(title="Normalized Value"),
    dragmode="pan",
)

fig.show()
