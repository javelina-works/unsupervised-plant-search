import cv2
import numpy as np

# Callback function for trackbar
def update_threshold(x):
    """
    This function is called whenever the trackbar value changes.
    It applies the thresholding and updates the displayed image.
    """
    # Get the current threshold value from the trackbar
    thresh_value = cv2.getTrackbarPos('Threshold', 'Thresholding GUI')
    
    # Apply thresholding
    _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Display the thresholded image
    cv2.imshow('Thresholding GUI', thresh_img)

# Load the TIFF file
# file_path = 'input\Bridle-Trail-RR3-6-2-2015-orthophoto.tif'
file_path = "input/ESPG-4326-orthophoto.tif"
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image!")
    exit()

# Create a window
cv2.namedWindow('Thresholding GUI', cv2.WINDOW_NORMAL)

# Create a trackbar to adjust threshold
cv2.createTrackbar('Threshold', 'Thresholding GUI', 0, 255, update_threshold)

# Display the initial image
cv2.imshow('Thresholding GUI', image)

# Wait indefinitely until the user presses the ESC key
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# Release all resources
cv2.destroyAllWindows()
