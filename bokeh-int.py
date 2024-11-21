from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.layouts import column, row

import rasterio
import numpy as np
from matplotlib import cm

# Step 1: Load GeoTIFF and extract bounds
tiff_file = "input/ESPG-4326-orthophoto.tif"  # Replace with your file path

with rasterio.open(tiff_file) as src:
    # Read RGB bands
    r = src.read(1).astype(float)
    g = src.read(2).astype(float)
    b = src.read(3).astype(float)

    # Normalize RGB bands for display
    r_norm = (r - np.min(r)) / (np.max(r) - np.min(r))
    g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
    b_norm = (b - np.min(b)) / (np.max(b) - np.min(b))

    # Combine into an RGBA image
    alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1).astype(float)
    rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))

    # Extract bounds for proper axis scaling
    bounds = src.bounds  # (left, bottom, right, top)


def calculate_index(index_name):
    """Calculate a vegetation index and return a colormap-applied RGBA image."""
    if index_name == "NDVI":
        index = (r_norm - g_norm) / (r_norm + g_norm + 1e-6)
    elif index_name == "VARI":
        index = (g_norm - r_norm) / (g_norm + r_norm - b_norm + 1e-6)
    elif index_name == "GNDVI":
        index = (g_norm - b_norm) / (g_norm + b_norm + 1e-6)
    else:  # Original image
        return rgb_image

    # Normalize the index for display
    index_norm = (index - np.min(index)) / (np.max(index) - np.min(index))

    # Apply a colormap (e.g., viridis)
    colormap = cm.get_cmap("viridis")
    colored_index = colormap(index_norm)[:, :, :3]  # Drop alpha channel

    return colored_index

# Step 2: Convert RGBA to uint8 for Bokeh and flip vertically
rgba_image = np.flipud((rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2]))

# Step 3: Create a Bokeh figure with proper axis bounds
p = figure(
    title="Interactive GeoTIFF Viewer",
    x_range=(bounds.left, bounds.right),
    y_range=(bounds.bottom, bounds.top),
    match_aspect=True,
    active_scroll="wheel_zoom",
    tools="pan,wheel_zoom,reset",  # Enable panning and zooming
    sizing_mode="scale_height",  # Adjust figure dimensions to viewport
)

# Step 4: Add the RGBA image to the plot
p.image_rgba(
    image=[rgba_image],
    x=bounds.left,
    y=bounds.bottom,
    dw=bounds.right - bounds.left,
    dh=bounds.top - bounds.bottom,
)

# Step 4: Create a dropdown for placeholder interaction
dropdown = Select(
    title="Select Option:",
    value="Original",
    options=["Original", "Option 1", "Option 2"],  # Placeholder options
)

# Step 4: Create a dropdown for placeholder interaction
dropdown1 = Select(
    title="Select Option:",
    value="Original",
    options=["Original", "Option 1", "Option 2"],  # Placeholder options
)

# Step 5: Create two placeholder sliders
slider1 = Slider(
    title="Placeholder Slider 1",
    start=0,
    end=100,
    value=50,
)

slider2 = Slider(
    title="Placeholder Slider 2",
    start=0,
    end=200,
    value=100,
)

# Step 6: Create another dropdown
dropdown2 = Select(
    title="Additional Option:",
    value="Default",
    options=["Default", "Option A", "Option B"],  # Placeholder options
)

# Step 7: Layout the widgets and figure
controls = column(dropdown1, slider1, slider2, dropdown2)
layout = row(p, controls, sizing_mode="stretch_both")
curdoc().add_root(layout)