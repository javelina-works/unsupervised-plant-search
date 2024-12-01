import numpy as np
from rasterio.io import MemoryFile
from matplotlib import cm

def process_geotiff(file_path):
    """Load and preprocess GeoTIFF."""
    with open(file_path, "rb") as f:
        decoded = f.read()
    with MemoryFile(decoded) as memfile:
        with memfile.open() as src:
            r, g, b = [src.read(i).astype(float) for i in range(1, 4)]
            r_norm, g_norm, b_norm = [
                (band - np.min(band)) / (np.max(band) - np.min(band))
                for band in (r, g, b)
            ]
            alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 1)
            rgba_image = np.dstack((r_norm, g_norm, b_norm, alpha))
            bounds = src.bounds
            return rgba_image, (r_norm, g_norm, b_norm), bounds

def calculate_index(index_name, bands, alpha, colormap="RdYlGn"):
    """Calculate vegetation index and return a normalized RGBA image."""
    r_norm, g_norm, b_norm = bands
    if index_name == "VARI":
        index = (g_norm - r_norm) / (g_norm + r_norm - b_norm + 1e-6)
    elif index_name == "GNDVI":
        index = (g_norm - b_norm) / (g_norm + b_norm + 1e-6)
    else:
        return None
    index = np.clip(index, -1, 1)
    index_norm = (index + 1) / 2
    colored_index = cm.get_cmap(colormap)(index_norm)
    colored_index[..., -1] = alpha
    return colored_index

def to_bokeh_rgba(image):
    """Convert a normalized RGBA image to uint32 for Bokeh."""
    return np.flipud((image * 255).astype(np.uint8).view(dtype=np.uint32).reshape(image.shape[:2]))

def compute_histogram(index_data):
    """Compute histogram of vegetation index."""
    hist, edges = np.histogram(index_data, bins=125, range=(-1, 1))
    return hist, edges
