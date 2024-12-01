from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, Select, Slider, Button, ColumnDataSource

from utils.logging_utils import setup_logger
import logging
from utils.geo_utils import process_geotiff, calculate_index, to_bokeh_rgba, compute_histogram
# from widgets import create_histogram_figures, create_image_figure, create_controls
from widgets.planner import create_image_figure, create_planner_column


# Initialize logger
logger = setup_logger(name="geo_tiff_app", log_level=logging.DEBUG)

# Initialize Data Sources
image_source = ColumnDataSource(data={"image": []})
# hist_source = ColumnDataSource(data={"x": [], "y": []})

# Initialize Global Variables
tiff_file = "input/MADRID_RGB.tif"

# Process initial data
logger.info("Processing initial GeoTIFF file.")
rgba_image, index_data, bounds = process_geotiff(tiff_file)
image_source.data = {"image": [to_bokeh_rgba(rgba_image)]}
# hist, edges = compute_histogram(index_data)
# midpoints = (edges[:-1] + edges[1:]) / 2
# hist_source.data = {"x": midpoints, "y": hist}

# Create figures
image_figure = create_image_figure(bounds, image_source)
# hist_figure, range_figure, spectrum_range = create_histogram_figures(hist_source)

# # Create controls
# controls = create_controls(
#     image_source=image_source,
#     hist_source=hist_source,
#     spectrum_range=spectrum_range,
#     rgba_image=rgba_image,
#     bounds=bounds,
# )

planner = create_planner_column(image_figure)


# Layout the widgets and figures
layout = column(planner, sizing_mode="stretch_both")
curdoc().add_root(layout)
logger.info("Application started.")
