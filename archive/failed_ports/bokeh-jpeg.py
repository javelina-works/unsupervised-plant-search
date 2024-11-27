from bokeh.models import ColumnDataSource, FileInput, Div, Range1d
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column
from PIL import Image
import base64
import io
import copy
import numpy as np

# Initialize data source and message div
image_source = ColumnDataSource(data={"image": []})
message = Div(text="Upload a JPEG image to display.", width=400, height=30)
# display_image = None

# Create the plot
p = figure(
    title="Uploaded JPEG Viewer",
    tools="pan,wheel_zoom,reset", 
    x_range=(0, 300), 
    y_range=(0, 300),
    # sizing_mode="scale_height",  # Adjust figure height to viewport height
)
display_image = p.image_rgba(
    image="image", 
    source=image_source,
    x=0, 
    y=0, 
    dw=300, 
    dh=300, 
)

# FileInput widget for uploading files
file_input = FileInput(accept=".jpg,.jpeg")

def process_jpeg(file_contents):
    """Process the uploaded JPEG file and update the image source."""
    try:
        # Handle Base64 header (if present)
        if "," in file_contents:
            header, encoded = file_contents.split(",", 1)
        else:
            encoded = file_contents  # Assume the entire content is Base64-encoded

        # Decode Base64 content
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded)).convert("RGBA")  # Convert to RGBA format

        # Convert the image to a NumPy array
        img_array = np.array(image)
        image_array = np.flipud(img_array)

        height, width, _ = image_array.shape
        rgba_image = image_array.view(dtype=np.uint32).reshape(height, width)

        image_source.data = {"image": [rgba_image]}

        # Update plot ranges to match the image dimensions
        # p.x_range = Range1d(0, width)
        # p.y_range = Range1d(0, height)
        # p.width = width
        # p.height = height

        # p.image_rgba(image="image", x=0, y=0, dw=width, dh=height, source=image_source)
        # p.sizing_mode = "scale_height"

        # image_source.data = {"image": [rgba_image], "dw": [width], "dh": [height]}
        
        # display_image.image = rgba_image

        message.text = "JPEG processed and displayed successfully!"
        return rgba_image
    except Exception as e:
        message.text = f"Error processing JPEG: {e}"
        raise

def upload_callback(attr, old, new):
    """Handle file upload."""
    file_contents = file_input.value
    if file_contents:
        message.text = "Processing file..."
        returned_image = process_jpeg(file_contents)
        # image_source.data = {"image": [returned_image]}

file_input.on_change("value", upload_callback)



# Layout
layout = column(file_input, message, p)
curdoc().add_root(layout)
