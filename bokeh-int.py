from bokeh.plotting import figure, show
import rasterio
import numpy as np

# Step 1: Load GeoTIFF and extract bounds
tiff_file = "input\ESPG-4326-orthophoto.tif"  # Replace with your file path

with rasterio.open(tiff_file) as src:
    # Read RGB bands
    r = src.read(1).astype(float)
    g = src.read(2).astype(float)
    b = src.read(3).astype(float)
    
    # Create an alpha channel: Fully transparent where all bands are zero
    alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1).astype(float)
    
    # Normalize bands for display
    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))
    
    # Combine bands into an RGBA image
    rgba_image = np.dstack((r, g, b, alpha))  # Include the alpha channel

    # Extract bounds for proper axis scaling
    bounds = src.bounds  # (left, bottom, right, top)

# Step 2: Convert RGBA to uint8 for Bokeh
rgba_image = (rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2])

# Step 3: Create a Bokeh figure with proper axis bounds
p = figure(
    title="Interactive GeoTIFF Viewer",
    x_range=(bounds.left, bounds.right),
    y_range=(bounds.bottom, bounds.top),
    match_aspect=True,
    tools="pan,wheel_zoom,reset",  # Enable panning and zooming
)

# Step 4: Add the RGBA image to the plot
p.image_rgba(
    image=[rgba_image],
    x=bounds.left,
    y=bounds.bottom,
    dw=bounds.right - bounds.left,
    dh=bounds.top - bounds.bottom,
)

# Step 5: Show the plot
show(p)  # Opens the interactive plot in your browser
