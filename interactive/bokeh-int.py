from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.layouts import layout
import rasterio
import numpy as np

# Step 1: Load GeoTIFF
tiff_file = "input\ESPG-4326-orthophoto.tif"

with rasterio.open(tiff_file) as src:
    r = src.read(1).astype(float)  # Red band
    g = src.read(2).astype(float)  # Green band
    b = src.read(3).astype(float)  # Blue band

    # Normalize the RGB bands for display
    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))

    # Combine the bands into an RGBA image
    rgb_image = np.dstack((r, g, b, np.ones_like(r)))  # Add alpha channel

# Step 2: Prepare RGBA image for Bokeh (convert to uint8)
rgba_image = (rgb_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgb_image.shape[:2])

# Step 3: Create a Bokeh plot
p = figure(
    title="Interactive GeoTIFF Viewer",
    x_range=(0, rgba_image.shape[1]),
    y_range=(0, rgba_image.shape[0]),
    match_aspect=True,
    tools="pan,wheel_zoom,reset",  # Enable pan and zoom
)

# Display the RGBA image
p.image_rgba(image=[rgba_image], x=0, y=0, dw=rgba_image.shape[1], dh=rgba_image.shape[0])

# Step 4: Output to the notebook or browser
# output_notebook()  # For Jupyter or interactive environments
show(p)  # This will open the plot in your default browser
