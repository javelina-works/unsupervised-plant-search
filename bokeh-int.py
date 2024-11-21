from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Slider, PointDrawTool
from bokeh.layouts import column, row
import rasterio
import numpy as np

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

# Step 2: Convert RGBA to uint32 for Bokeh and flip vertically
rgba_image = np.flipud((rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2]))

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
marker_source = ColumnDataSource(data={"x": [], "y": []})

# Add circle markers to the plot
p.circle(x="x", y="y", size=10, color="red", source=marker_source)

# Step 5: Add PointDrawTool for draggable markers
draw_tool = PointDrawTool(renderers=[p.renderers[-1]], empty_value="red")
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool  # Set PointDrawTool as the active tool

# Step 6: Add placeholder controls
dropdown1 = Select(title="Select Option:", value="Original", options=["Original", "Option 1", "Option 2"])
slider1 = Slider(title="Placeholder Slider 1", start=0, end=100, value=50)
slider2 = Slider(title="Placeholder Slider 2", start=0, end=200, value=100)
dropdown2 = Select(title="Additional Option:", value="Default", options=["Default", "Option A", "Option B"])

# Step 7: Layout the widgets and figure
controls = column(dropdown1, slider1, slider2, dropdown2)
layout = row(p, controls, sizing_mode="stretch_both")
curdoc().add_root(layout)
