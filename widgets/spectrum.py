from bokeh.models import Div, Select, Slider, RangeTool, ColumnDataSource, Button, CustomJS
from bokeh.plotting import figure
from bokeh.layouts import column
from functools import partial



def create_histogram_figures(hist_source):
    """Create histogram and range figures."""
    hist_figure = figure(title="Index Value Frequency", height=150)
    hist_figure.line(x="x", y="y", source=hist_source, line_width=2)

    range_figure = figure(height=130, y_range=hist_figure.y_range)
    range_tool = RangeTool(x_range=hist_figure.x_range)
    range_figure.add_tools(range_tool)
    return hist_figure, range_figure, range_tool

def create_controls(image_source, hist_source, spectrum_range, rgba_image, bounds, compute_histogram, calculate_index, to_bokeh_rgba):
    """Create interactive controls with callbacks."""

    # Dropdown for vegetation index
    view_select = Select(
        title="Select View:",
        value="Regular",
        options=["VARI", "GNDVI"],
    )

    # Dropdown for colormap
    color_select = Select(
        title="Colormap:",
        value="RdYlGn",
        options=["RdYlGn", "viridis", "plasma"],
    )

    # Range display
    range_display = Div(text="<b>Selected Range:</b> Start = -0.5, End = 0.5", width=400)

    # Slider example
    slider = Slider(title="Threshold", start=0, end=100, value=50)

    # Update image based on dropdown selection
    def update_view(attr, old, new):
        index_name = view_select.value
        new_image, new_index = calculate_index(index_name, rgba_image, alpha=rgba_image[..., -1])
        image_source.data = {"image": [to_bokeh_rgba(new_image)]}
        hist, edges = compute_histogram(new_index)
        hist_source.data = {"x": (edges[:-1] + edges[1:]) / 2, "y": hist}

    # Update colormap
    def update_colormap(attr, old, new):
        colormap_name = color_select.value
        new_image, _ = calculate_index(view_select.value, rgba_image, alpha=rgba_image[..., -1], colormap=colormap_name)
        image_source.data = {"image": [to_bokeh_rgba(new_image)]}

    # Attach callbacks
    view_select.on_change("value", update_view)
    color_select.on_change("value", update_colormap)

    # Button to save data (example)
    save_button = Button(label="Save", button_type="success")
    save_button.on_click(lambda: print("Save button clicked!"))

    # Organize controls into a column layout
    controls = column(view_select, color_select, slider, range_display, save_button)
    return controls
