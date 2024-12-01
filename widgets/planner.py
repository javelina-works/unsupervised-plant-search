from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import (
    CrosshairTool, TableColumn, DataTable, CustomJS,
    ColumnDataSource, PointDrawTool, Button, Div,
)


def create_image_figure(bounds, image_source):
    """Create the Bokeh figure for displaying the image."""
    p = figure(
        title="Interactive GeoTIFF Viewer",
        x_range=(bounds.left, bounds.right),
        y_range=(bounds.bottom, bounds.top),
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
    # p.output_backend = "webgl"
    return p

def create_planner_column(image_figure):

    crosshair = CrosshairTool()
    image_figure.add_tools(crosshair)

    # Div to display the coordinates
    # ==============================
    coords_display = Div(text="Mouse Coordinates: (x: --, y: --)", width=400, height=30)

    # CustomJS callback for updating coordinates
    callback = CustomJS(args=dict(coords=coords_display), code="""
        const {x, y} = cb_obj; // Get the mouse event from cb_obj
        // Update the Div text with the new coordinates
        coords.text = `Mouse Coordinates: (x: ${x.toFixed(7)}, y: ${y.toFixed(7)})`;
    """)

    # Attach the CustomJS to the plot's mouse move event
    image_figure.js_on_event("mousemove", callback)


    # Create a data source for draggable markers
    # ==================================================
    marker_source = ColumnDataSource(data={"x": [], "y": [], "label": []})
    points = image_figure.scatter(x="x", y="y", size=10, color="red", source=marker_source) # Add circle markers to the plot
    image_figure.line(x="x", y="y", source=marker_source, line_width=2, color="green")  # Line connecting points
    image_figure.text(x="x", y="y", text="label", source=marker_source, text_font_size="10pt", text_baseline="middle", color="yellow")

    draw_tool = PointDrawTool(renderers=[points], empty_value="1")
    image_figure.add_tools(draw_tool)
    image_figure.toolbar.active_tap = draw_tool  # Set PointDrawTool as the active tool


    # Button to save mission data to file
    # =====================================
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


    # DataTable to display clicked waypoints
    # ======================================
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
    marker_source.js_on_change('data', js_callback)


    # Organize interactive image into a column layout
    # ===============================================
    image_container = column(image_figure)
    image_container.sizing_mode = "stretch_both"

    data_col = column(coords_display, save_button, data_table)
    data_col.width = 400
    data_col.min_width = 400
    data_col.sizing_mode = "scale_height"

    planner_row = row(image_container, data_col)
    planner_row.sizing_mode = "stretch_both"
    return planner_row


