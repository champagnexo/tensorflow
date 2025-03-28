import tensorflow as tf
import numpy as np
import glob
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2

# Adjust image size
IMG_SIZE = 224
BATCH_SIZE = 16

# Define TFRecord parsing function (for positive samples)
def _parse_function(proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    
    example = tf.io.parse_single_example(proto, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalize

    label = tf.sparse.to_dense(example['image/object/class/label'], default_value=1)  # Default to 'Butt'

    label = label[0] if tf.size(label) > 0 else tf.constant(1, dtype=tf.int64)  # Ensure valid label

    return image, label

# Load positive samples (from TFRecord)
tfrecord_path = "datasets/train/cigarette-butts.tfrecord"
positive_dataset = (tf.data.TFRecordDataset([tfrecord_path])
                    .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE))

# Load negative samples (from generated images)
def load_negative_samples():
    images = []
    labels = []
    
    for img_path in glob.glob("datasets/generated_negatives/*.jpg"):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize
        images.append(img)
        labels.append(0)  # Label as 'Not a Butt'

    return tf.data.Dataset.from_tensor_slices((np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)))

negative_dataset = load_negative_samples()

# Combine datasets
full_dataset = (positive_dataset.concatenate(negative_dataset)
                .shuffle(buffer_size=1000)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))

num_samples = sum(1 for _ in full_dataset.unbatch())
print(f"Total samples: {num_samples}")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)

# Create model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
steps_per_epoch = num_samples // BATCH_SIZE
model.fit(full_dataset, epochs=10, steps_per_epoch=steps_per_epoch)

# Save model
model.save("cigarette_butt_classifier.h5")

print("Model training complete!")

# validate & test report
validation_tfrecord_path = "datasets/valid/cigarette-butts.tfrecord"

validation_dataset = (tf.data.TFRecordDataset([validation_tfrecord_path])
                      .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .prefetch(tf.data.experimental.AUTOTUNE))
val_loss, val_acc = model.evaluate(validation_dataset)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

test_tfrecord_path = "datasets/test/cigarette-butts.tfrecord"

test_dataset = (tf.data.TFRecordDataset([test_tfrecord_path])
                .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
