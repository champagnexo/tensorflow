import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("cigarette_detector.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open("cigarette_detector.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to cigarette_detector.tflite")
