import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab

def detect_blood_vessels(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Detect edges using Canny
    edges = cv2.Canny(enhanced, 50, 150)
    return edges

# Load and process the image
image = cv2.imread('/content/drive/MyDrive/B.E. Project/archive 2/train_path/AK/ISIC_0024468.jpg')  # Replace with the actual path to your image
vessel_edges = detect_blood_vessels(image)

# Display the result
cv2_imshow(vessel_edges)  # Use cv2_imshow instead of cv2.imshow
