import rasterio
import numpy as np
import matplotlib.pyplot as plt
import mpld3

# Vegetation Index Functions
def calculate_index(index_name, rgb_data):
    r, g, b = rgb_data
    if index_name == "NDVI":
        return (r - g) / (r + g + 1e-6)
    elif index_name == "VARI":
        return (g - r) / (g + r - b + 1e-6)
    elif index_name == "GNDVI":
        return (g - b) / (g + b + 1e-6)
    else:
        return np.moveaxis(rgb_data / np.max(rgb_data), 0, -1)

# Load GeoTIFF
geotiff_path = "input\ESPG-4326-orthophoto.tif"
with rasterio.open(geotiff_path) as src:
    r = src.read(1).astype(float)
    g = src.read(2).astype(float)
    b = src.read(3).astype(float)
    rgb_data = np.stack([r, g, b], axis=0)

# Initial plot
fig, ax = plt.subplots(figsize=(8, 6))
original_image = calculate_index("Original", rgb_data)
img = ax.imshow(original_image, interpolation="nearest")
ax.set_title("GeoTIFF with Vegetation Index Selection")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cbar = plt.colorbar(img, ax=ax, orientation="vertical", label="Value")

# JavaScript Dropdown
dropdown_html = """
<div style="text-align: center;">
    <label for="indexSelect">Select Vegetation Index:</label>
    <select id="indexSelect" onchange="updateImage()">
        <option value="Original">Original</option>
        <option value="NDVI">NDVI</option>
        <option value="VARI">VARI</option>
        <option value="GNDVI">GNDVI</option>
    </select>
</div>
"""

js_script = """
<script>
    function updateImage() {
        var index = document.getElementById("indexSelect").value;
        var imgElement = document.querySelector("image.mpld3_canvas");
        var dataMap = {
            "Original": "data_original", 
            "NDVI": "data_ndvi",
            "VARI": "data_vari", 
            "GNDVI": "data_gndvi"
        };
        imgElement.src = dataMap[index];
    }
</script>
"""

# Combine the plot and dropdown
html = mpld3.fig_to_html(fig)
html = html.replace("</body>", f"{dropdown_html}\n{js_script}\n</body>")

# Save and serve
with open("interactive_plot.html", "w") as f:
    f.write(html)

print("Interactive plot saved to 'interactive_plot.html'. Open this file in a browser.")
