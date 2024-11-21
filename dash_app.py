import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import rasterio
import base64
import io
import numpy as np
import plotly.express as px
from scipy import ndimage

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("GeoTIFF Image Upload and Vegetation Index (GRVI)"),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload GeoTIFF'),
        accept='.tif, .tiff',
    ),
    html.Div(id='image-info', style={'margin-top': '20px'}),
    dcc.Graph(id='grvi-graph', style={'margin-top': '20px'}),
    html.Div(id='bounding-box-info', style={'margin-top': '20px'}),
])

# Function to calculate GRVI (Green-Red Vegetation Index)
def calculate_grvi(red, green):
    return (green - red) / (green + red)

# Callback to process the uploaded file
@app.callback(
    [Output('image-info', 'children'),
     Output('grvi-graph', 'figure'),
     Output('bounding-box-info', 'children')],
    Input('upload-image', 'contents')
)
def update_image_info(contents):
    if contents is None:
        return "Upload a GeoTIFF file to view details.", {}, ""

    # Decode the base64 encoded image file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img_data = io.BytesIO(decoded)

    try:
        # Open the GeoTIFF file
        with rasterio.open(img_data) as src:
            # Read Red, Green, and Blue bands
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

            # Get image dimensions and number of bands
            image_dimensions = src.shape  # height, width
            num_bands = src.count

            # Calculate GRVI
            grvi = calculate_grvi(red, green)

            # Create GRVI figure using Plotly
            fig = px.imshow(grvi, color_continuous_scale='RdYlGn', title="Vegetation Index (GRVI)")
            fig.update_layout(
                xaxis_title="Width",
                yaxis_title="Height",
                coloraxis_colorbar_title="GRVI"
            )

            # For bounding box: Simple thresholding to detect vegetation areas
            threshold = 0.2  # Adjust threshold for detecting vegetation
            vegetation_mask = grvi > threshold
            bounding_boxes = []

            # Find bounding boxes (bounding box is min/max of the valid pixels in the mask)
            labeled, num_labels = ndimage.label(vegetation_mask)
            for i in range(1, num_labels + 1):
                y, x = np.where(labeled == i)
                ymin, ymax = y.min(), y.max()
                xmin, xmax = x.min(), x.max()
                bounding_boxes.append((xmin, ymin, xmax, ymax))

            bounding_boxes_info = f"Found {len(bounding_boxes)} bounding boxes for potential plants."

            return (
                f"Image Dimensions: {image_dimensions[0]} x {image_dimensions[1]}, Number of Bands: {num_bands}",
                fig,
                bounding_boxes_info
            )

    except Exception as e:
        return f"Error loading image: {str(e)}", {}, ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
