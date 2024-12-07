import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from google.colab.patches import cv2_imshow
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Paths
input_folder = '/content/drive/MyDrive/B.E. Project/archive 2/train_path/'
output_folder = '/content/drive/MyDrive/B.E. Project/extracted_imgs/'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function for feature extraction
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

# Function for blood vessel detection
def detect_blood_vessels(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Detect edges using Canny
    edges = cv2.Canny(enhanced, 50, 150)
    return edges

# Process each sub-folder and file
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # Full path to the current image
            file_path = os.path.join(subdir, file)
            image = cv2.imread(file_path)
            
            if image is None:
                continue  # Skip invalid images
            
            # Extract features
            features = extract_features(image)
            
            # Detect blood vessels
            vessel_edges = detect_blood_vessels(image)
            
            # Save features to CSV
            relative_subdir = os.path.relpath(subdir, input_folder)  # Keep sub-folder structure
            output_csv_dir = os.path.join(output_folder, relative_subdir)
            os.makedirs(output_csv_dir, exist_ok=True)
            
            # Save feature vector
            feature_file = os.path.join(output_csv_dir, file.split('.')[0] + '_features.csv')
            pd.DataFrame([features]).to_csv(feature_file, index=False)
            
            # Save vessel edge image
            output_vessel_image = os.path.join(output_csv_dir, file.split('.')[0] + '_vessels.jpg')
            cv2.imwrite(output_vessel_image, vessel_edges)

print("Processing complete! Results are saved in:", output_folder)
