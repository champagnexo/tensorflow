import cv2
import numpy as np
import os

output_dir = "datasets/generated_negatives"
os.makedirs(output_dir, exist_ok=True)

for i in range(100):  # Generate 100 images
    img = np.ones((224, 224, 3), dtype=np.uint8) * np.random.randint(230, 255)  # Light gray/white
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)  # Add slight noise
    noisy_img = cv2.add(img, noise)

    cv2.imwrite(f"{output_dir}/negative_{i}.jpg", noisy_img)

print("Generated negative samples successfully!")
