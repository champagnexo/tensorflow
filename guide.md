# **TIRAMISU – Raspberry Pi AI Camera Image Recognition (Fully Dockerized)**
Tiramisu is an **AI-powered image recognition project** that captures images using the **Raspberry Pi AI Camera** and classifies them with **TensorFlow Lite (TFLite)** inside a **Docker container**.

---

## **1️⃣ Install Docker & Docker Compose on Raspberry Pi**
### **Install Docker**
```sh
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker  # Refresh groups without logging out
docker --version  # Verify installation
```

### **Install Docker Compose**
```sh
sudo apt install -y libffi-dev libssl-dev
sudo apt install -y python3 python3-pip
sudo pip3 install docker-compose
docker compose --version  # Verify installation
```

---

## **2️⃣ Create the Tiramisu Project**
Navigate to your workspace and create the project folder:
```sh
mkdir tiramisu && cd tiramisu
```

---

## **3️⃣ Project Structure**
```
📂 tiramisu/
├── 📜 Dockerfile         # Builds the container
├── 📜 docker-compose.yml # Manages services
├── 📜 detect.py          # Main image recognition script
├── 📷 camera.py          # Handles image capture
├── 🤖 model.tflite       # TensorFlow Lite model (add your model here)
├── 📄 requirements.txt   # Python dependencies
└── 📝 README.md          # Documentation
```

---

## **4️⃣ Create the `Dockerfile`**
📄 **`Dockerfile`**: Defines the container environment.
```dockerfile
# Use a lightweight Python image
FROM python:3.11

# Install system dependencies for AI Camera and TensorFlow Lite
RUN apt update && apt install -y \
    libcamera0 libcamera-apps libcamera-dev python3-libcamera \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir opencv-python numpy tflite-runtime picamera2

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Run the detection script when the container starts
CMD ["python", "detect.py"]
```

---

## **5️⃣ Create the `docker-compose.yml`**
📄 **`docker-compose.yml`**: Manages the container.
```yaml
version: '3.8'

services:
  tiramisu:
    container_name: tiramisu-ai
    build: .
    devices:
      - "/dev/video0:/dev/video0"  # Allows access to the Raspberry Pi AI Camera
    volumes:
      - .:/app  # Mount local directory to container
    restart: unless-stopped
```

---

## **6️⃣ Create `requirements.txt`**
📄 **`requirements.txt`**: List of Python dependencies.
```
opencv-python
numpy
tflite-runtime
picamera2
```

---

## **7️⃣ Create `camera.py`**
📄 **`camera.py`**: Captures an image using the Raspberry Pi AI Camera.
```python
from picamera2 import Picamera2
import cv2

def capture_image(image_path="image.jpg"):
    """ Captures an image using the Raspberry Pi AI Camera and saves it. """
    picam2 = Picamera2()
    picam2.start()

    frame = picam2.capture_array()
    picam2.stop()

    cv2.imwrite(image_path, frame)
    print(f"Image saved to {image_path}")

    return image_path
```

---

## **8️⃣ Create `detect.py`**
📄 **`detect.py`**: Runs TensorFlow Lite inference on the captured image.
```python
import numpy as np
import tensorflow.lite as tflite
import cv2
from camera import capture_image

MODEL_PATH = "model.tflite"
IMAGE_PATH = "image.jpg"

def load_model():
    """ Loads the TensorFlow Lite model. """
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path):
    """ Loads and preprocesses the image for inference. """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Adjust to model's expected input size
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0  # Normalize
    return img

def predict():
    """ Captures an image, runs inference, and prints the result. """
    image_path = capture_image(IMAGE_PATH)
    interpreter = load_model()

    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    input_data = preprocess_image(image_path)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)  # Assuming classification model

    print(f"Prediction: {prediction} (Confidence: {np.max(output_data) * 100:.2f}%)")

if __name__ == "__main__":
    predict()
```

---

## **9️⃣ Start the Container**
Now, let’s **build and run** your Tiramisu AI system!

### **Build the Docker Image**
```sh
docker compose build
```

### **Run the Container**
```sh
docker compose up -d
```

### **View Logs (Check Outputs)**
```sh
docker logs -f tiramisu-ai
```

---

## **🔧 10️⃣ Troubleshooting & Fixes**
### **🔹 `ModuleNotFoundError: No module named 'libcamera'`**
Run:
```sh
sudo apt install -y libcamera0 libcamera-apps libcamera-dev python3-libcamera
```

### **🔹 `ValueError: Could not open 'model.tflite'`**
Make sure `model.tflite` exists in the project directory:
```sh
ls -lah $(pwd)/model.tflite
```

---

## **🚀 11️⃣ Auto-Start Tiramisu on Boot**
To **ensure Tiramisu starts on boot**, use:
```sh
sudo systemctl enable docker
```

Or, add this to **crontab**:
```sh
crontab -e
```
And add:
```
@reboot cd /home/pi/tiramisu && docker-compose up -d
```

---

## **✅ Summary**
✔ **Fully Dockerized** TensorFlow Lite project  
✔ **Captures images** using the Raspberry Pi AI Camera  
✔ **Runs inference** with TensorFlow Lite  
✔ **Manages everything using Docker Compose**  
✔ **Starts automatically on boot**  

---

## **🌟 Next Steps**
- Train your own **TensorFlow Lite model** and replace `model.tflite`.  
- Send predictions to a **remote server** or **dashboard**.  
- Add a **web UI** to display the results in real-time.  
