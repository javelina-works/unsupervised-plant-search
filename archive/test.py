from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, RangeTool
from bokeh.layouts import column, row
import rasterio
import numpy as np
from matplotlib import cm

# Step 1: Load GeoTIFF and preprocess
tiff_file = "input/ESPG-4326-orthophoto.tif"  # Replace with your file path

with rasterio.open(tiff_file) as src:
    r = src.read(1).astype(float)
    g = src.read(2).astype(float)
    b = src.read(3).astype(float)

    r_norm = (r - np.min(r)) / (np.max(r) - np.min(r))
    g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
    b_norm = (b - np.min(b)) / (np.max(b) - np.min(b))

    alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1).astype(float)
    rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))
    bounds = src.bounds

# Step 2: Define vegetation indices
def calculate_index(index_name, colormap_name="viridis"):
    """Calculate vegetation index and apply the chosen colormap."""
    if index_name == "VARI":
        index = (g_norm - r_norm) / (g_norm + r_norm - b_norm + 1e-6)
    elif index_name == "GNDVI":
        index = (g_norm - b_norm) / (g_norm + b_norm + 1e-6)
    else:
        return rgba_image

    index_clipped = np.clip(index, -1, 1)
    index_norm = (index_clipped + 1) / 2

    colormap = cm.get_cmap(colormap_name)
    colored_index = colormap(index_norm)
    colored_index[..., -1] = alpha
    return colored_index

def to_bokeh_rgba(image):
    """Convert an RGBA image to a uint32 array for Bokeh."""
    return np.flipud((image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(image.shape[:2]))

# Step 3: Prepare initial data
current_colormap = "viridis"
current_index = "Regular"
initial_image = calculate_index(current_index, current_colormap)
image_source = ColumnDataSource(data={"image": [to_bokeh_rgba(initial_image)]})

# Step 4: Create the main figure
p = figure(
    title="Interactive GeoTIFF Viewer",
    x_range=(bounds.left, bounds.right),
    y_range=(bounds.bottom, bounds.top),
    match_aspect=True,
    tools="pan,wheel_zoom,reset",
    sizing_mode="scale_height",
)
p.image_rgba(
    image="image",
    source=image_source,
    x=bounds.left,
    y=bounds.bottom,
    dw=bounds.right - bounds.left,
    dh=bounds.top - bounds.bottom,
)

# Step 5: Dropdown for view selection
dropdown1 = Select(
    title="Select View:",
    value="Regular",
    options=["Regular", "VARI", "GNDVI"],
)

def update_image(attr, old, new):
    """Update the displayed image when the view changes."""
    global current_index
    current_index = dropdown1.value
    new_image = calculate_index(current_index, current_colormap)
    image_source.data = {"image": [to_bokeh_rgba(new_image)]}

dropdown1.on_change("value", update_image)

# Step 6: Dropdown for colormap selection
dropdown2 = Select(
    title="Select Colormap:",
    value="viridis",
    options=["viridis", "plasma", "inferno", "cividis", "magma"],
)

def update_colormap(attr, old, new):
    """Update the colormap of the image."""
    global current_colormap
    current_colormap = dropdown2.value
    new_image = calculate_index(current_index, current_colormap)
    image_source.data = {"image": [to_bokeh_rgba(new_image)]}

dropdown2.on_change("value", update_colormap)

# Step 7: Layout and render
layout = column(p, dropdown1, dropdown2)
curdoc().add_root(layout)
