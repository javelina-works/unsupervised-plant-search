from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, Select, Slider, PointDrawTool, 
    RangeTool, Range1d, Div, DataTable, TableColumn, 
    CustomJS, Button, CrosshairTool, FileInput
)
from bokeh.layouts import column, row
import base64
from rasterio.io import MemoryFile
import numpy as np
from matplotlib import cm
import os
import logging
from utils.logging_utils import setup_logger

# Initialize the logger
logger = setup_logger(name="my_project_logger", log_level=logging.DEBUG)


# Step 1: Load GeoTIFF and preprocess
# tiff_file = "input/ESPG-4326-orthophoto.tif"  # Replace with your file path
tiff_file = "input/MADRID_RGB.tif"  # Replace with your file path

image_source = ColumnDataSource(
    data={"image": []}, 
    # default_values={"image": "input/ESPG-4326-orthophoto.tif"},
)
upload_message = Div(text="Upload a GeoTIFF to display.", width=400, height=30)

# FileInput widget
file_input = FileInput(accept=".tif,.tiff")

def fix_base64_padding(data: str) -> str:
    """Fix padding for Base64 strings."""
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return data

def process_geotiff(file_contents):
    """Process the uploaded GeoTIFF and update the plot."""

    global r_norm, g_norm, b_norm, non_transparent_mask, rgba_image, alpha, bounds

    decoded = None  # To hold the decoded or raw data
    # file_contents = fix_base64_padding(file_contents) # Fix padding before decoding

    # Handle local file or uploaded file
    if os.path.isfile(file_contents):
        logger.debug("Loaded GeoTIFF from local file.")
        with open(file_contents, "rb") as f:
            decoded = f.read()

    elif "," in file_contents:
        logger.debug("Uploaded file with Base64 header.")
        _, encoded = file_contents.split(",", 1)
        decoded = base64.b64decode(encoded)

    else:
        logger.debug("Uploaded file without Base64 header.")
        decoded = base64.b64decode(file_contents)

    # Memory read files
    # https://rasterio.readthedocs.io/en/latest/topics/memory-files.html#memoryfile-bytesio-meets-namedtemporaryfile
    with MemoryFile(decoded) as memfile:
        with memfile.open() as src:
            # Read RGB bands
            r = src.read(1).astype(float)
            g = src.read(2).astype(float)
            b = src.read(3).astype(float)

            # Normalize RGB bands
            r_norm = (r - np.min(r)) / (np.max(r) - np.min(r))
            g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
            b_norm = (b - np.min(b)) / (np.max(b) - np.min(b))

            # Combine into an RGBA image
            alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1).astype(float)
            non_transparent_mask = alpha > 0  # True for pixels that are not fully transparent

            rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))
            # rgba_image = np.flipud((rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2]))
            bounds = src.bounds  # Extract bounds for proper axis scaling

            # Update the ColumnDataSource
            # image_source.data = {"image": [rgba_image]}

            # return rgba_image


# Callback for file upload
def upload_callback(attr, old, new):
    """Handle file upload."""
    file_contents = file_input.value
    if file_contents:
        process_geotiff(file_contents)  # Remove header and decode

# DEBUGGING
def inspect_index(index):
    print("Index count:", np.size(index))
    print("Index min:", np.nanmin(index))
    print("Index max:", np.nanmax(index))
    print("Index mean:", np.nanmean(index))
    print("Index median:", np.nanmedian(index))
    print("Index std:", np.nanstd(index))

# DEBUGGING
def count_out_of_bounds_values(data, lower=-1, upper=1):
    """Count the number of values outside the specified range."""
    # Identify values outside the range
    out_of_bounds_mask = (data < lower) | (data > upper)
    out_of_bounds_count = np.sum(out_of_bounds_mask)
    print(f"Number of values outside [-1, 1]: {out_of_bounds_count}")
    
    # Count the number of out-of-bounds values
    total_valid = np.sum(~np.isnan(data))
    print(f"Number of values inside [-1, 1]: {total_valid}")
    
    # Calculate percentage
    if total_valid > 0:
        percentage_out_of_bounds = (out_of_bounds_count / total_valid) * 100
        print(f"Percentage of out-of-bounds values: {percentage_out_of_bounds:.2f}%")

# Define vegetation index calculations
def calculate_index(index_name, colormap_name="RdYlGn", lower_clip=None, upper_clip=None):
    """Calculate vegetation index and return a normalized image."""
    if index_name == "VARI":
        index = (g_norm - r_norm) / (g_norm + r_norm - b_norm + 1e-6)
    elif index_name == "GNDVI":
        index = (g_norm - b_norm) / (g_norm + b_norm + 1e-6)
    else:  # Default to RGB for the regular view
        return rgba_image, None

    # Normalize the index to [-1, 1] for visualization
    index[~non_transparent_mask] = np.nan  # Set transparent regions to NaN
    index_clipped = np.clip(index, -1, 1)  # Ensure values are in the range [-1, 1]

    # Apply a colormap (e.g., viridis)
    # =================================
    if lower_clip is not None:
        index[index < lower_clip] = -1
    if upper_clip is not None:
        index[index > upper_clip] = 1  

    index_norm = (index + 1) / 2  # Normalize to [0, 1] for colormap    

    colormap = cm.get_cmap(colormap_name)
    colored_index = colormap(index_norm)  # Returns RGBA values (0-1)
    colored_index[..., -1] = alpha  # Apply original transparency mask

    return colored_index, index_clipped

def to_bokeh_rgba(image):
    """Convert an RGBA array (float) to a uint32 array for Bokeh."""
    return np.flipud((image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(image.shape[:2]))


# Step 3: Prepare initial data and Bokeh components
# logger.debug(f"Initializing default image: {tiff_file}")
# logger.debug(f"File exists: {os.path.isfile(tiff_file)}")
# logger.debug(f"Current working directory: {os.getcwd()}")
process_geotiff(tiff_file)
initial_colormap = "RdYlGn"
initial_image, initial_index = calculate_index("VARI")
image_source = ColumnDataSource(data={"image": [to_bokeh_rgba(initial_image)]})
logger.debug("Initial image prepared.")


# Compute histogram
def compute_histogram(index_values):
    """Compute a histogram of index values."""
    if index_values is None:
        return np.array([]), np.array([])
    hist, edges = np.histogram(index_values, bins=125, range=(-1, 1))
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
    tools="pan,wheel_zoom,reset",  # Enable panning and zooming
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
p.output_backend = "webgl"
crosshair = CrosshairTool()
p.add_tools(crosshair)

# Div to display the coordinates
coords = Div(text="Mouse Coordinates: (x: --, y: --)", width=300, height=30)

# CustomJS callback for updating coordinates
callback = CustomJS(args=dict(coords=coords), code="""
    const {x, y} = cb_obj; // Get the mouse event from cb_obj
    // Update the Div text with the new coordinates
    coords.text = `Mouse Coordinates: (x: ${x.toFixed(7)}, y: ${y.toFixed(7)})`;
""")

# Attach the CustomJS to the plot's mouse move event
p.js_on_event("mousemove", callback)


# Step 5: Single Histogram as a Line Graph
midpoints = (edges[:-1] + edges[1:]) / 2  # Compute midpoints of bins
line_hist_source = ColumnDataSource(data={"x": midpoints, "y": hist})  # Source for the line graph

midpoints_center = ((max(midpoints) + min(midpoints))/2)
low_midpoint = min(midpoints) + (midpoints_center - min(midpoints))/2 
high_midpoint = max(midpoints) - (max(midpoints) - midpoints_center)/2

hist_figure = figure(
    title="Index Value Frequency",
    height=150, width=800,
    # x_range=Range1d(-1, 1),
    x_range=Range1d(low_midpoint, high_midpoint),
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

        new_image, new_index = calculate_index(
            view_select.value, color_select.value,
            lower_clip=range_start, upper_clip=range_end
        )
        image_source.data = {"image": [to_bokeh_rgba(new_image)]}

# Attach the callback to the RangeTool's x_range
spectrum_range.x_range.on_change("start", update_range)
spectrum_range.x_range.on_change("end", update_range)
# spectrum_range.on_change("x_range", update_range)



# Step 4: Create a data source for draggable markers
marker_source = ColumnDataSource(data={"x": [], "y": [], "label": []})
points = p.scatter(x="x", y="y", size=10, color="red", source=marker_source) # Add circle markers to the plot
p.line(x="x", y="y", source=marker_source, line_width=2, color="green")  # Line connecting points
p.text(x="x", y="y", text="label", source=marker_source, text_font_size="10pt", text_baseline="middle", color="yellow")

draw_tool = PointDrawTool(renderers=[points], empty_value="1")
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool  # Set PointDrawTool as the active tool

# DataTable to display clicked points
columns = [
    TableColumn(field="label", title="Waypoint #"),
    TableColumn(field="x", title="X Coordinate"),
    TableColumn(field="y", title="Y Coordinate"),
]
data_table = DataTable(source=marker_source, columns=columns, width=400, height=280)

# CustomJS to number points incrementally
# Good: Faster to handle this in the browser
js_callback = CustomJS(args=dict(source=marker_source), code="""
    const data = source.data;
    const labels = data['label'];
    for (let i = 0; i < data['x'].length; i++) {
        labels[i] = (i + 1).toString();  // Incremental numbering starts from 1
    }
    source.change.emit();  // Trigger update
""")

# Attach the CustomJS to the data source
marker_source.js_on_change('data', js_callback)

# Button to save data to file
save_button = Button(label="Save to File", button_type="success")

# Callback to save data to file
def save_to_file():
    """Save the current DataTable values to a waypoints file."""
    data = marker_source.data  # Get the data from the source

    waypoints_filename = 'gen2.waypoints'
    with open(waypoints_filename, 'w') as f:
        # Write header for the MAVLink file (QGroundControl WPL version)
        f.write("QGC WPL 110\n")  # Write header
        
        # Check if there's any data to write
        num_points = len(data["x"])  # Number of rows
        if num_points == 0:
            print("No points to save!")
            return

        # Add home point (index 0)
        # Home point command = 3, current WP = 1
        f.write(f"0\t1\t0\t3\t0\t0\t0\t0\t{data['y'][0]}\t{data['x'][0]}\t100.000000\t1\n")

        # Add waypoints starting from index 1 (regular waypoints)
        for index in range(1, num_points):
            # Waypoint command = 16
            f.write(f"{index}\t0\t0\t16\t0\t0\t0\t0\t{data['y'][index]}\t{data['x'][index]}\t100.000000\t1\n")

    print(f"Waypoints have been exported to {waypoints_filename}")

save_button.on_click(save_to_file)


# Create a dropdown for toggling views
view_select = Select(
    title="Select View:",
    value="VARI",
    options=["Regular", "VARI", "GNDVI"],  # Add vegetation indices as options
)

def update_image(attr, old, new):
    """Update the displayed image based on the selected view."""
    new_image, new_index = calculate_index(view_select.value, color_select.value, spectrum_range.x_range.start)
    image_source.data = {"image": [to_bokeh_rgba(new_image)]}

    # Update histogram
    hist, edges = compute_histogram(new_index)
    midpoints = (edges[:-1] + edges[1:]) / 2
    line_hist_source.data = {"x": midpoints, "y": hist}  # Update line graph source

view_select.on_change("value", update_image)

# Goes with file_input object
# Moved here for scoping bc using globals like a dumbass
file_input.on_change("value", upload_callback, update_image)

# Step 6: Dropdown for colormap selection
color_select = Select(
    title="Select Colormap:",
    value="RdYlGn",
    options=["RdYlGn", "Spectral", "viridis", "plasma", "inferno", "magma", "cividis", "jet" ],
)

def update_colormap(attr, old, new):
    """Update the colormap of the image."""
    new_image, new_index = calculate_index(view_select.value, color_select.value, spectrum_range.x_range.start)
    image_source.data = {"image": [to_bokeh_rgba(new_image)]}

color_select.on_change("value", update_colormap)


# Placeholder controls
slider1 = Slider(title="Placeholder Slider 1", start=0, end=100, value=50)
slider2 = Slider(title="Placeholder Slider 2", start=0, end=200, value=100)

# Step 7: Layout the widgets and figure
controls = column(coords, file_input, view_select, color_select, hist_figure, range_figure, range_display, slider1, slider2, save_button, data_table)
layout = row(p, controls, sizing_mode="stretch_both")
curdoc().add_root(layout)
