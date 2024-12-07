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

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import frangi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# Convert the image to grayscale if it's in color
if len(image.shape) > 2:
    image = color.rgb2gray(image)

# Apply Frangi filter using scikit-image
# The Frangi filter is applied to the image to highlight vessels
frangi_image = frangi(image)

# Alternatively, you can manually apply Hessian matrix and eigenvalue calculation
# Calculate Hessian matrix of the image
hessian = hessian_matrix(image, sigma=1.0)

# Get the eigenvalues of the Hessian matrix
eigenvalues = hessian_matrix_eigvals(hessian)

# The Frangi filter combines the eigenvalues to detect vessel-like structures
# We are interested in the ratio between the eigenvalues (typically, one eigenvalue is much smaller than the other for tubular structures)
vesselness = np.exp(- (eigenvalues[0]**2 + eigenvalues[1]**2) / (2.0 * 1.0**2))

# Display the original image and the processed image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(vesselness, cmap='hot')
ax[1].set_title('Vesselness (Frangi Filter)')
ax[1].axis('off')

plt.show()


# Load and process the image
image = cv2.imread('/content/drive/MyDrive/B.E. Project/archive 2/train_path/AK/ISIC_0024468.jpg')  # Replace with the actual path to your image
vessel_edges = detect_blood_vessels(image)

# Display the result
cv2_imshow(vessel_edges)  # Use cv2_imshow instead of cv2.imshow
