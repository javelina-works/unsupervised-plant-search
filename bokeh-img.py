import base64
import rasterio
import numpy as np
from bokeh.models import ColumnDataSource, FileInput, Div
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column

# Initialize data source and message div
image_source = ColumnDataSource(data={"image": []})
message = Div(text="Upload a TIFF image to display.", width=400, height=30)

# Create the plot
p = figure(title="Uploaded TIFF Viewer", tools="pan,wheel_zoom,reset", x_range=(0, 10), y_range=(0, 10))
p.image_rgba(image="image", x=0, y=0, dw=10, dh=10, source=image_source)

# FileInput widget for uploading files
file_input = FileInput(accept=".tif,.tiff")

def process_tiff(file_contents):
    """Process the uploaded TIFF file and update the image source."""
    try:
        # Decode Base64 content
        header, encoded = file_contents.split(",", 1)
        decoded = base64.b64decode(encoded)

        # Save the decoded content to a temporary file
        temp_filename = "/tmp/uploaded.tif"
        with open(temp_filename, "wb") as f:
            f.write(decoded)

        # Open the TIFF file and extract RGB bands
        with rasterio.open(temp_filename) as src:
            r = src.read(1).astype(float)
            g = src.read(2).astype(float)
            b = src.read(3).astype(float)

            # Normalize bands for RGBA image
            r_norm = (r - np.min(r)) / (np.max(r) - np.min(r))
            g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
            b_norm = (b - np.min(b)) / (np.max(b) - np.min(b))
            alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1).astype(float)

            # Combine into RGBA image
            rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))
            rgba_image = (rgba_image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(rgba_image.shape[:2])

            # Update ColumnDataSource
            image_source.data = {"image": [rgba_image]}

            # Adjust plot ranges to match TIFF bounds
            bounds = src.bounds
            p.x_range.start = bounds.left
            p.x_range.end = bounds.right
            p.y_range.start = bounds.bottom
            p.y_range.end = bounds.top

        message.text = "Image successfully uploaded and displayed."
    except Exception as e:
        message.text = f"Error processing TIFF: {e}"

def upload_callback(attr, old, new):
    """Handle file upload."""
    file_contents = file_input.value
    if file_contents:
        message.text = "Processing file..."
        process_tiff(file_contents)

file_input.on_change("value", upload_callback)

# Layout
layout = column(file_input, message, p)
curdoc().add_root(layout)
