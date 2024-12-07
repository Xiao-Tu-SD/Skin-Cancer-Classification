# Import Libraries
import os
import cv2
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from mahotas.features import haralick
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import pickle

# Handcrafted Features (Haralick + Blood Vessel Detection)
def extract_handcrafted_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    haralick_features = haralick(image).mean(axis=0)
    vessel_edges = cv2.Canny(image, 50, 150)
    vessel_features = np.sum(vessel_edges)
    combined_features = np.hstack([haralick_features, vessel_features])
    return combined_features

# Deep Features (Pre-trained EfficientNetB0)
def extract_deep_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    deep_features = model.predict(img_array)
    return deep_features.flatten()

# Process Dataset for Feature Extraction
def process_dataset(dataset_dir, model):
    handcrafted_features = []
    deep_features = []
    labels = []

    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(subdir, file)
                handcrafted = extract_handcrafted_features(image_path)
                handcrafted_features.append(handcrafted)
                deep = extract_deep_features(image_path, model)
                deep_features.append(deep)
                labels.append(os.path.basename(subdir))
    handcrafted_features = np.array(handcrafted_features)
    deep_features = np.array(deep_features)
    labels = np.array(labels)
    return handcrafted_features, deep_features, labels

# Predict on New Images
def predict_image(image_path, classifier, base_model, label_encoder, scaler):
    handcrafted = extract_handcrafted_features(image_path)
    handcrafted = scaler.transform([handcrafted])
    deep = extract_deep_features(image_path, base_model)
    fused = np.hstack([handcrafted, deep])
    prediction = classifier.predict(fused)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Main Function
if __name__ == "__main__":
    # Paths
    dataset_dir = '/content/drive/MyDrive/B.E. Project/archive 2/train_path/'
    model_save_path = '/content/drive/MyDrive/B.E. Project/classification_model.h5'
    label_encoder_path = '/content/drive/MyDrive/B.E. Project/label_encoder.pkl'
    scaler_path = '/content/drive/MyDrive/B.E. Project/feature_scaler.pkl'

    # Load Pre-trained CNN
    base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

    # Process Dataset
    print("Extracting features...")
    handcrafted_features, deep_features, labels = process_dataset(dataset_dir, base_model)

    # Standardize Handcrafted Features
    scaler = StandardScaler()
    handcrafted_features = scaler.fit_transform(handcrafted_features)

    # Fuse Features
    fused_features = np.hstack([handcrafted_features, deep_features])

    # Encode Labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Define Classifier Model
    classifier = Sequential([
        Dense(128, activation='relu', input_dim=fused_features.shape[1]),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(np.unique(encoded_labels)), activation='softmax')
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the Model
    print("Training the model...")
    classifier.fit(fused_features, encoded_labels, epochs=50, batch_size=32)

    # Save the Model and Encoders
    print("Saving model and encoders...")
    classifier.save(model_save_path)
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Predict on a New Image
    test_image_path = '/path/to/new/image.jpg'
    predicted_class = predict_image(test_image_path, classifier, base_model, label_encoder, scaler)
    print(f"Predicted Class: {predicted_class}")
