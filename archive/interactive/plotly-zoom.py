import rasterio
import numpy as np
import plotly.express as px

# Step 1: Load the GeoTIFF file
tiff_file = "input\ESPG-4326-orthophoto.tif"

with rasterio.open(tiff_file) as src:
    # Read RGB bands (adjust based on your file's band structure)
    r = src.read(1).astype(float)
    g = src.read(2).astype(float)
    b = src.read(3).astype(float)

    # Normalize bands for display
    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))

    # Stack into an RGB image
    rgb_image = np.stack([r, g, b], axis=-1)

# Step 2: Create an interactive Plotly figure
fig = px.imshow(rgb_image, color_continuous_scale=None, origin="lower")
fig.update_layout(
    title="Interactive GeoTIFF Viewer",
    dragmode="pan",  # Enable panning
    xaxis=dict(title="Longitude", constrain="domain"),
    yaxis=dict(title="Latitude", scaleanchor="x", constrain="domain"),
)

# Step 3: Show the plot
fig.show()
