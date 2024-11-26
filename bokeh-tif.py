from bokeh.models import ColumnDataSource, FileInput, Div, Range1d
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column
from PIL import Image
import base64
import io
import copy
import numpy as np

import rasterio
from rasterio.io import MemoryFile

# Initialize data source and message div
image_source = ColumnDataSource(data={"image": []})
message = Div(text="Upload a TIF image to display.", width=400, height=30)
# display_image = None

# Create the plot
p = figure(
    title="Uploaded TIF Viewer",
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
file_input = FileInput(accept=".tif,.tiff")

def process_tiff(file_contents):
    """Process the uploaded JPEG file and update the image source."""
    try:
        # Handle Base64 header (if present)
        if "," in file_contents:
            print("Header found!")
            header, encoded = file_contents.split(",", 1)
        else:
            encoded = file_contents  # Assume the entire content is Base64-encoded
            print("No header yet found!")

        # Decode Base64 content
        decoded = base64.b64decode(encoded)
        # image = Image.open(io.BytesIO(decoded)).convert("RGBA")  # Convert to RGBA format

        image_array = None

        with MemoryFile(decoded) as memfile:
            with memfile.open() as dataset:
                image = dataset.read()

                # Convert the image to a NumPy array
                image_array = np.flipud(np.array(image))

        # print(image_array)

                r = dataset.read(1)
                g = dataset.read(2)
                b = dataset.read(3)

                # Normalize bands to 0-255
                def normalize(band):
                    return (255 * (band - band.min()) / (band.max() - band.min())).astype(np.uint8)

                r_norm, g_norm, b_norm = map(normalize, [r, g, b])
                
                alpha = np.full_like(r_norm, 255, dtype=np.uint8)
                # alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1).astype(float)
                # non_transparent_mask = alpha > 0  # True for pixels that are not fully transparent

                rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))

                # rgba_image = np.flipud((image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(image.shape[:2]))
                rgba_image = rgba_image.view(dtype=np.uint32).reshape(r.shape[0], r.shape[1])

                height, width, _ = image_array.shape
                # rgba_image = image_array.view(dtype=np.uint32).reshape(height, width)

                image_source.data = {"image": [rgba_image]}

                # Update plot ranges
                height, width = r.shape
                p.x_range = Range1d(0, width)
                p.y_range = Range1d(0, height)
                p.width = width
                p.height = height


        print("Success processing image!")
        message.text = "TIF processed and displayed successfully!"
        return rgba_image
    
    except Exception as e:
        message.text = f"Error processing TIF: {e}"
        raise

def upload_callback(attr, old, new):
    """Handle file upload."""
    print("Upload file callback triggered!")
    file_contents = file_input.value
    if file_contents:
        message.text = "Processing file..."
        process_tiff(file_contents)
        # image_source.data = {"image": [returned_image]}
    else:
        print("No file contents found!")

file_input.on_change("value", upload_callback)


# Layout
layout = column(file_input, message, p)
curdoc().add_root(layout)
