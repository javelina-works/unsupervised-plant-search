import streamlit as st
from bokeh.models import (
    ColumnDataSource, Div, Range1d, CrosshairTool, PointDrawTool, CustomJS,
    TableColumn, DataTable
)
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from PIL import Image
import base64
import numpy as np
from rasterio.io import MemoryFile
import pandas as pd


# Page title and description
st.title("Manual Targeting")
st.write("Upload an image and interact with it in the Bokeh plot.")


# Initialize data source and message div
image_source = ColumnDataSource(data={"image": []})
message = Div(text="Upload a TIF image to display.", width=400, height=30)
# display_image = None

# Create the plot
p = figure(
    title="Uploaded TIF Viewer",
    x_range=(0, 300), 
    y_range=(0, 300),
    match_aspect=True,
    active_scroll="wheel_zoom",
    tools="pan,wheel_zoom,reset",  # Enable panning and zooming
    sizing_mode="scale_height",  # Adjust figure height to viewport height
)
p.image_rgba(
    image="image", 
    source=image_source,
    x=0, 
    y=0, 
    dw=300, 
    dh=300, 
)

p.output_backend = "webgl"
crosshair = CrosshairTool()
p.add_tools(crosshair)


# Add Support for draggable markers
marker_source = ColumnDataSource(data={"x": [], "y": [], "label": []})
points = p.scatter(x="x", y="y", size=10, color="red", source=marker_source) # Add circle markers to the plot
p.line(x="x", y="y", source=marker_source, line_width=2, color="green")  # Line connecting points
p.text(x="x", y="y", text="label", source=marker_source, text_font_size="10pt", text_baseline="middle", color="yellow")

draw_tool = PointDrawTool(renderers=[points], empty_value="1")
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool  # Set PointDrawTool as the active tool

# DataTable to display clicked points

# fake_source = pd.DataFrame(columns=["x", "y", "label"])
# columns = [
#     TableColumn(field="label", title="Waypoint #"),
#     TableColumn(field="x", title="X Coordinate"),
#     TableColumn(field="y", title="Y Coordinate"),
# ]
# data_table = DataTable(source=fake_source, columns=columns, width=400, height=280)


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


def process_tiff(file_contents):
    """Process the uploaded JPEG file and update the image source."""
    try:
        # print(file_contents)
        # # Handle Base64 header (if present)
        # if "," in file_contents:
        #     print("Header found!")
        #     header, encoded = file_contents.split(",", 1)
        # else:
        #     encoded = file_contents  # Assume the entire content is Base64-encoded
        #     print("No header yet found!")
    
        # decoded = base64.b64decode(encoded) # Decode Base64 content

        # Memory read files
        # https://rasterio.readthedocs.io/en/latest/topics/memory-files.html#memoryfile-bytesio-meets-namedtemporaryfile
        with MemoryFile(file_contents) as memfile:
            with memfile.open() as src:
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
                rgba_image = np.flipud((rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2]))
                bounds = src.bounds  # Extract bounds for proper axis scaling

                image_source.data = {"image": [rgba_image]}

        print("Success processing image!")
        message.text = "TIF processed and displayed successfully!"
        return rgba_image
    
    except Exception as e:
        message.text = f"Error processing TIF: {e}"
        raise



# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    image_rgba = process_tiff(bytes_data)

# Display the Bokeh plot
st.bokeh_chart(p)

st.write("### Waypoints Table")
if st.button("Update Marker Data"):
    # Force Streamlit to rerun and display updated marker_source
    st.write(marker_source.data)

# st.bokeh_chart(data_table)
