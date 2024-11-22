from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Slider, PointDrawTool
from bokeh.layouts import column, row
import rasterio
import numpy as np
from matplotlib import cm

# Step 1: Load GeoTIFF and preprocess
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


# Define vegetation index calculations
def calculate_index(index_name):
    """Calculate vegetation index and return a normalized image."""
    if index_name == "VARI":
        index = (g_norm - r_norm) / (g_norm + r_norm - b_norm + 1e-6)
    elif index_name == "GNDVI":
        index = (g_norm - b_norm) / (g_norm + b_norm + 1e-6)
    else:  # Default to RGB for the regular view
        return rgba_image

    # Normalize the index to [0, 1] for visualization
    index_norm = (index - np.min(index)) / (np.max(index) - np.min(index))

    # Apply a colormap (e.g., viridis)
    colormap = cm.get_cmap("viridis")
    colored_index = colormap(index_norm)[:, :, :3]  # Remove alpha channel

    return colored_index

def to_bokeh_rgba(image):
    """Convert an RGB array (float) to a uint32 array for Bokeh."""
    flipped = np.flipud((image * 255).astype(np.uint8))  # Flip vertically for Bokeh
    r, g, b = flipped[..., 0], flipped[..., 1], flipped[..., 2]
    a = np.full_like(r, 255, dtype=np.uint8)  # Fully opaque alpha channel
    return (a.astype(np.uint32) << 24) | (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)


# Step 2: Convert RGBA to uint32 for Bokeh and flip vertically
rgba_image = np.flipud((rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2]))

# Step 3: Prepare initial data and Bokeh components
# initial_image = to_bokeh_rgba(rgb_image)
image_source = ColumnDataSource(data={"image": [rgba_image]})

# Step 3: Create a Bokeh figure
p = figure(
    title="Interactive GeoTIFF Viewer with Draggable Markers",
    x_range=(bounds.left, bounds.right),
    y_range=(bounds.bottom, bounds.top),
    match_aspect=True,
    active_scroll="wheel_zoom",
    tools="pan,wheel_zoom,reset",  # Enable panning and zooming
    sizing_mode="scale_height",  # Adjust figure height to viewport height
)

# Add the RGBA image to the plot
p.image_rgba(
    image=[rgba_image],
    x=bounds.left,
    y=bounds.bottom,
    dw=bounds.right - bounds.left,
    dh=bounds.top - bounds.bottom,
)

# Step 4: Create a data source for draggable markers
# marker_source = ColumnDataSource(data={"x": [], "y": []})
# p.circle(x="x", y="y", size=10, color="red", source=marker_source) # Add circle markers to the plot
# draw_tool = PointDrawTool(renderers=[p.renderers[-1]], empty_value="red")
# p.add_tools(draw_tool)
# p.toolbar.active_tap = draw_tool  # Set PointDrawTool as the active tool

# Create a dropdown for toggling views
view_select = Select(
    title="Select View:",
    value="Regular",
    options=["Regular", "VARI", "GNDVI"],  # Add vegetation indices as options
)

def update_image(attr, old, new):
    """Update the displayed image based on the selected view."""
    index_name = view_select.value
    new_image = calculate_index(index_name)
    # bokeh_image = to_bokeh_rgba(new_image)
    image_source.data = {"image": [new_image]}

view_select.on_change("value", update_image)

# Placeholder controls
slider1 = Slider(title="Placeholder Slider 1", start=0, end=100, value=50)
slider2 = Slider(title="Placeholder Slider 2", start=0, end=200, value=100)
dropdown2 = Select(title="Additional Option:", value="Default", options=["Default", "Option A", "Option B"])

# Step 7: Layout the widgets and figure
controls = column(view_select, slider1, slider2, dropdown2)
layout = row(p, controls, sizing_mode="stretch_both")
curdoc().add_root(layout)
