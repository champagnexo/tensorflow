import os
import time
import datetime
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np
import cv2
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.models import Model

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

# Load a pretrained model
base_model = EfficientNetB0(weights="imagenet", include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.output)

# 1. Log Model Architecture
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
model_summary_str = "\n".join(model_summary)

with file_writer.as_default():
    tf.summary.text("Model Architecture", model_summary_str, step=0)

# 2. Log Model Visualization
dot_img_file = "/tmp/pretrained_model.png"
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

with file_writer.as_default():
    tf.summary.image("Model Visualization", tf.expand_dims(tf.io.decode_png(tf.io.read_file(dot_img_file)), 0), step=0)

# 3. Log Weights Histogram
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# (Optional) Dummy Training to Log Histograms
dummy_x = tf.random.normal((32, 224, 224, 3))
dummy_y = tf.random.uniform((32,), maxval=1000, dtype=tf.int32)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(dummy_x, dummy_y, epochs=1, callbacks=[tensorboard_callback])

print(f"TensorBoard logs are saved at: {log_dir}")

# Function to classify and log images
def classify_and_log_image(image_path, step):
    # Preprocess for the model
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Normalize pixel values

    # Predict
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]  # Top-1 prediction
    predicted_class = decoded_predictions[1]  # Class name
    confidence = decoded_predictions[2]  # Confidence score

    # Annotate image
    label = f"{predicted_class}: {confidence*100:.2f}%"
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.putText(image_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Convert for TensorBoard
    digit = tf.convert_to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), dtype=tf.uint8)
    digit = tf.expand_dims(digit, 0)  # Add batch dimension

    # Log image and prediction
    with file_writer.as_default():
        tf.summary.image(f"Classified Image {step}", digit, step=step)
        tf.summary.text(f"Prediction {step}", f"{predicted_class}: {confidence*100:.2f}%", step=step)

    print(f"Processed: {image_path} -> {predicted_class} ({confidence*100:.2f}%)")

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
        
        os.system('libcamera-still --width 1536 --height 1024 -o img/input/smaller.jpg')

# Start the monitoring process
monitor_directory()