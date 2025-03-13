# Raspberry Pi AI Camera - TensorFlow Lite Image Recognition


## 🎯 Project Overview
This project runs **image recognition** on a **Raspberry Pi** using a **TensorFlow Lite (TFLite) model** and a Raspberry Pi Camera. The system captures an image and classifies it in real time!

🔹 **Technologies Used:**
- 🐍 Python
- 📷 Raspberry Pi Camera with `picamera2`
- ⚡ TensorFlow Lite (`tflite-runtime`)
- 🐳 Docker (Optional, for containerization)

---

## 🚀 Setup & Installation

### 1️⃣ **Install Dependencies**
Ensure your Raspberry Pi has all necessary dependencies:

```sh
sudo apt update && sudo apt install -y libcamera0 libcamera-apps libcamera-dev python3-libcamera
pip install opencv-python numpy tflite-runtime picamera2
```

### 2️⃣ **Clone the Repository**

```sh
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 3️⃣ **Run the Python Script**

```sh
python detect.py
```

This will capture an image and run inference using your `model.tflite` file.

---

## 🐳 Running in Docker (Optional)

### **Build & Run the Docker Container**
If you want to run this in a Docker container:

```sh
docker build -t pi-ai-camera .
docker run --rm -it \
  --device /dev/video0 \
  -v $(pwd):/app \
  pi-ai-camera
```

This ensures your **Python scripts & model files update without rebuilding** the Docker image.

---

## 📜 Project Structure
```
📂 your-repo/
├── 📜 detect.py       # Main script for capturing & classifying images
├── 🤖 model.tflite    # TensorFlow Lite model
├── 📷 camera.py       # Handles image capture
├── 📄 requirements.txt # Python dependencies
└── 📝 README.md       # This file
```

---

## 🏆 Features
✅ Capture images from the **Raspberry Pi Camera**  
✅ Run **real-time image recognition** using **TFLite**  
✅ **Lightweight & Fast** - Perfect for **Edge AI applications**  
✅ Optional **Docker support** for easier deployment  

---

## 🤖 Example Output
```sh
Image captured!
Running TensorFlow Lite inference...
Prediction: 🚗 Car (98.7% confidence)
```

---

## 🛠️ Troubleshooting
### 🔹 `ModuleNotFoundError: No module named 'libcamera'`
Run:
```sh
sudo apt install -y libcamera0 libcamera-apps libcamera-dev python3-libcamera
```

### 🔹 `ValueError: Could not open 'model.tflite'`
Ensure your `model.tflite` file exists in the correct location.
Run:
```sh
ls -lah $(pwd)/model.tflite
```

---

## 📜 License
This project is **open-source** under the [MIT License](LICENSE).
