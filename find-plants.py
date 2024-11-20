import rasterio
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


# Where is your image?
geotiff_path = "input\ESPG-4326-orthophoto.tif"  # Replace with your file path

# Load the Geotiff
def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        orthophoto = src.read()  # Read all bands
        profile = src.profile   # Save geotiff profile (metadata)
    return orthophoto, profile

# Compute NDVI
def compute_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)  # Prevent division by zero
    return ndvi

# Segment Image using K-Means Clustering
def segment_ndvi(ndvi, n_clusters=2):
    flat_ndvi = ndvi.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(flat_ndvi)
    segmented_image = labels.reshape(ndvi.shape)
    return segmented_image

# Morphological Filtering to Clean Noise
def clean_segmentation(segmented_image, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_image = cv2.morphologyEx(segmented_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return cleaned_image

# Find Contours and Extract Bounding Boxes
def find_bounding_boxes(cleaned_image):
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return bounding_boxes

# Overlay Bounding Boxes on Original Image
def overlay_bounding_boxes(image, bounding_boxes):
    image_copy = image.copy()
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
    return image_copy

# Save Bounding Boxes as CSV
def save_bounding_boxes(bounding_boxes, output_path):
    df = pd.DataFrame(bounding_boxes, columns=["x", "y", "width", "height"])
    df.to_csv(output_path, index=False)

# Main Execution
if __name__ == "__main__":
    output_csv = "bounding_boxes.csv"
    orthophoto, profile = load_geotiff(geotiff_path)

    # Assume NIR is band 4 and Red is band 1 (update indices if needed)
    nir_band = orthophoto[3]  # Adjust based on band arrangement in your data
    red_band = orthophoto[0]

    # Step 1: Compute NDVI
    ndvi = compute_ndvi(nir_band, red_band)

    # Step 2: Segment NDVI using K-Means
    segmented_image = segment_ndvi(ndvi)

    # Step 3: Clean Segmentation with Morphological Filtering
    cleaned_image = clean_segmentation(segmented_image)

    # Step 4: Find Bounding Boxes
    bounding_boxes = find_bounding_boxes(cleaned_image)

    # Step 5: Overlay Bounding Boxes on Original Image
    overlay_image = overlay_bounding_boxes(orthophoto[0], bounding_boxes)  # Using Red band for visualization

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image, cmap="gray")
    plt.title("Detected Plants with Bounding Boxes")
    plt.axis("off")
    plt.show()

    # Save bounding boxes to CSV
    save_bounding_boxes(bounding_boxes, output_csv)
    print(f"Bounding boxes saved to {output_csv}")
