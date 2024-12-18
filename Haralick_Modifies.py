{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "14fvoA78LTLnEQ_omqvSpwC9_bsCtiXeC",
      "authorship_tag": "ABX9TyMXJmRTpH2gJOd9JKpG9zDy",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Xiao-Tu-SD/Skin-Cancer-Classification/blob/main/Haralick_Modifies.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "print(os.path.exists('/content/drive/MyDrive/Skin cancer classification/Train_path/AK/ISIC_0028393.jpg'))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "72aZ1QwS55vF",
        "outputId": "564f5615-0c5a-4cd7-e701-f5498e51ce87"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "True\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Riddhi code for refernces"
      ],
      "metadata": {
        "id": "Qh2jJsujdWmE"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "qKBce4WC3S2n",
        "outputId": "eb3062fa-0fb0-4a62-b75b-e6e545bdaa56",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'entropy': 115394.38025934197, 'contrast': 43.710517273576095, 'correlation': 0.9191522100666146, 'homogeneity': 0.22269187329107848, 'energy': 0.03195640112844987, 'mean_hue': 7.682066666666667, 'mean_saturation': 81.67766296296297, 'mean_value': 192.62321481481482}\n"
          ]
        }
      ],
      "source": [
        "import cv2\n",
        "import numpy as np\n",
        "from skimage.feature import graycomatrix, graycoprops\n",
        "\n",
        "def extract_features(image):\n",
        "    features = {}\n",
        "\n",
        "    # Convert to grayscale\n",
        "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
        "\n",
        "    # Entropy (Texture)\n",
        "    entropy = -np.sum(gray / 255.0 * np.log2(gray / 255.0 + 1e-6))\n",
        "    features['entropy'] = entropy\n",
        "\n",
        "    # Haralick Features\n",
        "    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)\n",
        "    features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]\n",
        "    features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]\n",
        "    features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]\n",
        "    features['energy'] = graycoprops(glcm, 'energy')[0, 0]\n",
        "\n",
        "    # Color Features (RGB/HSV Mean)\n",
        "    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)\n",
        "    features['mean_hue'] = np.mean(hsv[:, :, 0])\n",
        "    features['mean_saturation'] = np.mean(hsv[:, :, 1])\n",
        "    features['mean_value'] = np.mean(hsv[:, :, 2])\n",
        "\n",
        "    return features\n",
        "\n",
        "# Example usage\n",
        "image = cv2.imread('/content/drive/MyDrive/Skin cancer classification/Train_path/AK/ISIC_0028393.jpg')\n",
        "features = extract_features(image)\n",
        "print(features)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "er5IvNVXxtM1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import cv2## OpenCV library for image processing\n",
        "import numpy as np# Import the NumPy library for numerical operations\n",
        "from skimage.feature import graycomatrix, graycoprops  # Correct spelling of graycomatrix # Functions for GLCM and feature extraction\n",
        "from skimage.color import rgb2gray # Converts RGB images to grayscale\n",
        "from skimage import img_as_ubyte# Converts image to 8-bit unsigned integers\n",
        "from skimage.io import imread# Reads the image\n",
        "##Skimage is provides tools for image conversion, GLCM computation, and feature extraction."
      ],
      "metadata": {
        "id": "4Nbq4DwJ7oWL"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def haralick_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):##extract Haralick features from an image\n",
        "    \"\"\"\n",
        "    Extract enhanced Haralick texture features from an input image.\n",
        "\n",
        "    Parameters:\n",
        "        image (array): Input image (grayscale or RGB).\n",
        "        distances (list): List of pixel pair distances for GLCM computation.\n",
        "        angles (list): List of angles in radians for GLCM computation.\n",
        "\n",
        "    Returns:\n",
        "        features (dict): Dictionary of aggregated Haralick features.\n",
        "    \"\"\"\n",
        "    # Ensure image is grayscale\n",
        "    ##If the image has 3 channels (RGB), it is converted to grayscale using rgb2gray. This simplifies computations since GLCM requires grayscale input.\n",
        "    if len(image.shape) == 3:\n",
        "        image = rgb2gray(image)\n",
        "        image = img_as_ubyte(image)  # Convert to 8-bit for GLCM computation\n",
        "\n",
        "    # Compute GLCM\n",
        "    ##Creates a matrix (glcm) showing how often pixel pairs with specific relationships (distance & angle) appear together.\n",
        "    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)\n",
        "\n",
        "    # Initialize feature dictionary\n",
        "    haralick_features = {}\n",
        "\n",
        "    # Extract standard Haralick features\n",
        "    feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']\n",
        "    for feature_name in feature_names:\n",
        "        values = graycoprops(glcm, feature_name)\n",
        "        haralick_features[f'{feature_name}_mean'] = np.mean(values)#Average value across distances and angles.\n",
        "        haralick_features[f'{feature_name}_std'] = np.std(values)#Variability in the feature.\n",
        "        haralick_features[f'{feature_name}_range'] = np.ptp(values)# Difference between maximum and minimum values.\n",
        "\n",
        "    # Add custom features (e.g., entropy)measuring the randomness of the texture.\n",
        "    glcm_entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))# Calculates entropy to measure texture randomness\n",
        "    haralick_features['entropy'] = glcm_entropy# Stores the calculated entropy in the feature dictionary\n",
        "\n",
        "    return haralick_features"
      ],
      "metadata": {
        "id": "zqxqIGOs8tIf"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the specific image\n",
        "image_path = '/content/drive/MyDrive/Skin cancer classification/Train_path/AK/ISIC_0028393.jpg'\n",
        "image = imread(image_path)\n"
      ],
      "metadata": {
        "id": "SLPoL590821U"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Extract Haralick features\n",
        "features = haralick_features(image)\n",
        "\n",
        "# Display the features\n",
        "print(\"Extracted Haralick Features:\")\n",
        "for key, value in features.items():# Iterates through the dictionary and prints each key-value pair\n",
        "    print(f\"{key}: {value}\")# Prints each feature name (key) and its corresponding value"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wBPibNKN85g2",
        "outputId": "50c01727-8643-47c5-b8d0-44dc402433bf"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Extracted Haralick Features:\n",
            "contrast_mean: 15.362718959303606\n",
            "contrast_std: 7.239888348230214\n",
            "contrast_range: 20.64488926361238\n",
            "dissimilarity_mean: 2.5000114394734925\n",
            "dissimilarity_std: 0.6674100395434444\n",
            "dissimilarity_range: 1.8811913778744145\n",
            "homogeneity_mean: 0.36815048506164705\n",
            "homogeneity_std: 0.06920314670088903\n",
            "homogeneity_range: 0.2035848434856049\n",
            "ASM_mean: 0.0018260019397818648\n",
            "ASM_std: 0.00047574367763117024\n",
            "ASM_range: 0.0014382290499386188\n",
            "energy_mean: 0.04237180892037623\n",
            "energy_std: 0.005534595612778732\n",
            "energy_range: 0.016513069667904758\n",
            "correlation_mean: 0.9729328939762057\n",
            "correlation_std: 0.01278895475964443\n",
            "correlation_range: 0.03621001232347343\n",
            "entropy: 116.71893767886218\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Path to the directory containing AK class images\n",
        "image_dir = \"/content/drive/MyDrive/Skin cancer classification/Train_path/AK\""
      ],
      "metadata": {
        "id": "ApAafioRH28I"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "hoU5ba-dlIU9"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}