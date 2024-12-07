import cv2
import numpy as np
from google.colab.patches import cv2_imshow  

def detect_blood_vessels(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    
    edges = cv2.Canny(enhanced, 50, 150)
    return edges

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import frangi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

if len(image.shape) > 2:
    image = color.rgb2gray(image)

frangi_image = frangi(image)


hessian = hessian_matrix(image, sigma=1.0)

eigenvalues = hessian_matrix_eigvals(hessian)


vesselness = np.exp(- (eigenvalues[0]**2 + eigenvalues[1]**2) / (2.0 * 1.0**2))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(vesselness, cmap='hot')
ax[1].set_title('Vesselness (Frangi Filter)')
ax[1].axis('off')

plt.show()



image = cv2.imread('/content/drive/MyDrive/B.E. Project/archive 2/train_path/AK/ISIC_0024468.jpg')  
vessel_edges = detect_blood_vessels(image)


cv2_imshow(vessel_edges)
