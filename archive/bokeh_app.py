from bokeh.io import curdoc
from bokeh.models import FileInput, Paragraph
from io import BytesIO
import rasterio

# Function to load and process the uploaded GeoTIFF file
def load_and_process_file(file_content):
    global image_info
    
    try:
        # Convert the uploaded file to a BytesIO object
        img_data = BytesIO(file_content)
        
        # Try opening the GeoTIFF file
        with rasterio.open(img_data) as src:
            # Get image dimensions and number of bands
            image_dimensions = src.shape
            num_bands = src.count

            # Update the text in the app
            image_info.text = f"Image Dimensions: {image_dimensions}, Number of Bands: {num_bands}"
            print(f"Image Dimensions: {image_dimensions}, Number of Bands: {num_bands}")  # Debugging step
            
    except Exception as e:
        # If there is an error, print it in the terminal and update the app with the error message
        image_info.text = f"Error loading image: {str(e)}"
        print(f"Error loading image: {str(e)}")  # Debugging step

# Bokeh widget for file input
file_input = FileInput(accept=".tif,.tiff")
file_input.on_change("value", lambda attr, old, new: load_and_process_file(new))

# Paragraph widget to display image information or errors
image_info = Paragraph(text="Upload a GeoTIFF file to view image details.")

# Layout for the Bokeh app
layout = column(file_input, image_info)

# Add the layout to the current Bokeh document
curdoc().add_root(layout)
