from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Slider, PointDrawTool, RangeTool, Range1d, Div
from bokeh.layouts import column, row
import rasterio
import numpy as np
from matplotlib import cm
import sys

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
    non_transparent_mask = alpha > 0  # True for pixels that are not fully transparent

    rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))
    bounds = src.bounds  # Extract bounds for proper axis scaling

# Define vegetation index calculations
def calculate_index(index_name, colormap_name="RdYlGn"):
    """Calculate vegetation index and return a normalized image."""
    if index_name == "VARI":
        index = (g_norm - r_norm) / (g_norm + r_norm - b_norm + 1e-6)
    elif index_name == "GNDVI":
        index = (g_norm - b_norm) / (g_norm + b_norm + 1e-6)
    else:  # Default to RGB for the regular view
        return rgba_image, None

    index[~non_transparent_mask] = np.nan  # Set transparent regions to NaN

    # After calculating index_clipped in calculate_index
    print("Index min:", np.min(index))
    print("Index max:", np.max(index))
    print("Index mean:", np.mean(index))
    print("Index std:", np.std(index))


    # Normalize the index to [-1, 1] for visualization
    index_clipped = np.clip(index, -1, 1)  # Ensure values are in the range [-1, 1]
    index_norm = (index_clipped + 1) / 2  # Normalize to [0, 1] for colormap

    # Apply a colormap (e.g., viridis)
    colormap = cm.get_cmap(colormap_name)
    colored_index = colormap(index_norm)  # Returns RGBA values (0-1)
    colored_index[..., -1] = alpha  # Apply original transparency mask

    return colored_index, index_clipped

def to_bokeh_rgba(image):
    """Convert an RGBA array (float) to a uint32 array for Bokeh."""
    return np.flipud((image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(image.shape[:2]))


# Step 3: Prepare initial data and Bokeh components
initial_colormap = "RdYlGn"
initial_image, initial_index = calculate_index("VARI")
image_source = ColumnDataSource(data={"image": [to_bokeh_rgba(initial_image)]})

# Compute histogram
def compute_histogram(index_values):
    """Compute a histogram of index values."""
    if index_values is None:
        return np.array([]), np.array([])
    hist, edges = np.histogram(index_values, bins=50, range=(-1, 1))
    return hist, edges

hist, edges = compute_histogram(initial_index)
hist_source = ColumnDataSource(data={"top": hist, "left": edges[:-1], "right": edges[1:]})


# [!] Works great! Leave this one alone
# Step 3: Create a Bokeh figure
p = figure(
    title="Interactive GeoTIFF Viewer with Draggable Markers",
    x_range=(bounds.left, bounds.right),
    y_range=(bounds.bottom, bounds.top),
    match_aspect=True,
    active_scroll="wheel_zoom",
    tools="pan,wheel_zoom,reset,tap",  # Enable panning and zooming
    sizing_mode="scale_height",  # Adjust figure height to viewport height
)

# Add the RGBA image to the plot
p.image_rgba(
    image="image",
    source=image_source,
    x=bounds.left,
    y=bounds.bottom,
    dw=bounds.right - bounds.left,
    dh=bounds.top - bounds.bottom,
)


# Step 5: Single Histogram as a Line Graph
midpoints = (edges[:-1] + edges[1:]) / 2  # Compute midpoints of bins
line_hist_source = ColumnDataSource(data={"x": midpoints, "y": hist})  # Source for the line graph

hist_figure = figure(
    title="Index Value Frequency",
    height=150, width=800,
    x_range=Range1d(-1, 1),
    toolbar_location=None,
    tools="reset",
    background_fill_color="#efefef",
)

# Add the line and circles to represent the histogram
hist_figure.line(
    x="x", y="y",
    source=line_hist_source,
    line_width=2,
    color="blue",
)

range_figure = figure(
    height=130, width=800, 
    y_range=hist_figure.y_range,
    # x_axis_type="datetime", 
    y_axis_type=None,
    tools="", toolbar_location=None, 
    background_fill_color="#efefef"
)
range_figure.line(x="x", y="y", source=line_hist_source, line_width=2,
    color="blue")

# RangeTool to adjust histogram range
spectrum_range = RangeTool(x_range=hist_figure.x_range)
spectrum_range.overlay.fill_color = "green"
spectrum_range.overlay.fill_alpha = 0.2
range_figure.add_tools(spectrum_range)

# Div widget to display selected range
range_display = Div(
    text=f"<b>Selected Range:</b> Start = -0.5, End = 0.5",
    width=hist_figure.width, height=30,
)

# Callback to capture the selected range
def update_range(attr, old, new):
    """Capture the selected range from the RangeTool."""
    range_start = spectrum_range.x_range.start
    range_end = spectrum_range.x_range.end
    if range_start is not None and range_end is not None:
        range_display.text = (
            f"<b>Selected Range:</b> Start = {range_start:.2f}, "
            f"End = {range_end:.2f}"
        )

# Attach the callback to the RangeTool's x_range
spectrum_range.x_range.on_change("start", update_range)
spectrum_range.x_range.on_change("end", update_range)
spectrum_range.on_change("x_range", update_range)



# Step 4: Create a data source for draggable markers
# marker_source = ColumnDataSource(data={"x": [], "y": []})
# p.circle(x="x", y="y", size=10, color="red", source=marker_source) # Add circle markers to the plot
# draw_tool = PointDrawTool(renderers=[p.renderers[-1]], empty_value="red")
# p.add_tools(draw_tool)
# p.toolbar.active_tap = draw_tool  # Set PointDrawTool as the active tool

# Create a dropdown for toggling views
view_select = Select(
    title="Select View:",
    value="VARI",
    options=["Regular", "VARI", "GNDVI"],  # Add vegetation indices as options
)

def update_image(attr, old, new):
    """Update the displayed image based on the selected view."""
    new_image, new_index = calculate_index(view_select.value)
    image_source.data = {"image": [to_bokeh_rgba(new_image)]}

    # Update histogram
    hist, edges = compute_histogram(new_index)
    # hist_source.data = {"top": hist, "left": edges[:-1], "right": edges[1:]}
    midpoints = (edges[:-1] + edges[1:]) / 2
    line_hist_source.data = {"x": midpoints, "y": hist}  # Update line graph source

view_select.on_change("value", update_image)


# Step 6: Dropdown for colormap selection
color_select = Select(
    title="Select Colormap:",
    value="RdYlGn",
    options=["RdYlGn", "Spectral", "viridis", "plasma", "inferno", "magma", "cividis", "jet" ],
)

def update_colormap(attr, old, new):
    """Update the colormap of the image."""
    new_image, new_index = calculate_index(view_select.value, color_select.value)
    image_source.data = {"image": [to_bokeh_rgba(new_image)]}

color_select.on_change("value", update_colormap)


# Placeholder controls
slider1 = Slider(title="Placeholder Slider 1", start=0, end=100, value=50)
slider2 = Slider(title="Placeholder Slider 2", start=0, end=200, value=100)

# Step 7: Layout the widgets and figure
controls = column(view_select, color_select, hist_figure, range_figure, range_display, slider1, slider2)
layout = row(p, controls, sizing_mode="stretch_both")
curdoc().add_root(layout)
