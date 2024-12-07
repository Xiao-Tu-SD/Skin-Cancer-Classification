import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    features = {}

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Entropy (Texture)
    entropy = -np.sum(gray / 255.0 * np.log2(gray / 255.0 + 1e-6))
    features['entropy'] = entropy

    # Haralick Features
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['energy'] = graycoprops(glcm, 'energy')[0, 0]

    # Color Features (RGB/HSV Mean)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features['mean_hue'] = np.mean(hsv[:, :, 0])
    features['mean_saturation'] = np.mean(hsv[:, :, 1])
    features['mean_value'] = np.mean(hsv[:, :, 2])

    return features

# Example usage
image = cv2.imread('/content/drive/MyDrive/B.E. Project/archive 2/train_path/AK/ISIC_0024468.jpg')
features = extract_features(image)
print(features)
