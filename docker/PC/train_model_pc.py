import pandas as pd
import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8  # Adjust batch size for memory management

# Path to the dataset on your PC
dataset_path = r"C:\Users\Cristian\Desktop\archive\images"

# Function to load images and their bounding boxes
def load_images_from_csv(df, images_folder):
    images, labels, bboxes = [], [], []

    for _, row in df.iterrows():
        image_path = os.path.join(images_folder, row["filename"])
        img = cv2.imread(image_path)

        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE) / 255.0

            xmin = row["xmin"] / img.shape[1]
            ymin = row["ymin"] / img.shape[0]
            xmax = row["xmax"] / img.shape[1]
            ymax = row["ymax"] / img.shape[0]

            images.append(img)
            labels.append(row["class"])
            bboxes.append([xmin, ymin, xmax, ymax])

    return np.array(images), np.array(labels), np.array(bboxes)

# Load CSV files
train_df = pd.read_csv(os.path.join(dataset_path, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(dataset_path, "test_labels.csv"))

train_images, train_labels, train_bboxes = load_images_from_csv(train_df, os.path.join(dataset_path, "train"))
test_images, test_labels, test_bboxes = load_images_from_csv(test_df, os.path.join(dataset_path, "test"))

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Print dataset info
print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}, Train bboxes: {train_bboxes.shape}")
print(f"Test images: {test_images.shape}, Test labels: {test_labels.shape}, Test bboxes: {test_bboxes.shape}")

# Load MobileNetV2 model (lighter model)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze the pre-trained layers

# Build custom model on top of MobileNetV2
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(set(train_labels)), activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=BATCH_SIZE, validation_data=(test_images, test_labels))

# Save the model as .h5 file
model.save("cigarette_detector.h5")
print("Model saved as cigarette_detector.h5")
