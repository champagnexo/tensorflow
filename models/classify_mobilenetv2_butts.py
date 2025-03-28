import os
import time
import datetime
import shutil
import tensorflow as tf
import io
import numpy as np
import cv2
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

# Adjust the image size to match MobileNetV2's input (224x224)
IMG_SIZE = 224

# Define paths
input_path = "img/input"
processed_path = "img/processed"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Cleanup
os.system('rm -rf ./logs/')

# Ensure directories exist
os.makedirs(input_path, exist_ok=True)
os.makedirs(processed_path, exist_ok=True)

# TensorBoard writer
file_writer = tf.summary.create_file_writer(log_dir)

# Define class names and model
class_names = ['Null', 'Butt']
#model = tf.keras.models.load_model('"model/cigarette_butt_classifier.keras"')
model = tf.keras.models.load_model("model/cigarette_butt_classifier.h5")

# Function to classify and log images
def classify_and_log_image(image_path, step):
    # Load original image for visualization
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    # model was trained on TensorFlow datasets (which use RGB), swap color channels
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Preprocess for the model
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')  # Match training size
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image_array)
    if predictions.size == 0:
        print(f"Warning: No valid prediction for {image_path}")
        return
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Resize for annotation
    resized_image = cv2.resize(original_image, (200, 200), interpolation=cv2.INTER_AREA)
    label = f"{class_names[predicted_class]}: {confidence*100:.2f}%"
    cv2.putText(resized_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Convert for TensorBoard
    digit = tf.convert_to_tensor(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), dtype=tf.uint8)
    digit = tf.expand_dims(digit, 0)  # Add batch dimension

    # Log image and prediction
    with file_writer.as_default():
        tf.summary.image(f"Classified Image {step}", digit, step=step)
        tf.summary.text(f"Prediction {step}", f"{class_names[predicted_class]}: {confidence*100:.2f}%", step=step)

    print(f"Processed: {image_path} -> {class_names[predicted_class]} ({confidence*100:.2f}%)")

# Monitor and process new images
def monitor_directory():
    step = 0
    while True:
        # List all files in the input directory
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        
        for file_name in files:
            file_path = os.path.join(input_path, file_name)

            # Process the image
            try:
                classify_and_log_image(file_path, step)
                step += 1
                # Move the file to the processed directory
                shutil.move(file_path, os.path.join(processed_path, file_name))
            except IndexError as e:
                print(f"Error processing {file_name}: Index out of range. Check class mapping or predictions. Details: {e}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        # Wait before checking again
        time.sleep(2)

# Start the monitoring process
monitor_directory()
