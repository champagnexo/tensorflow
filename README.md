# Image Classification and Logging with TensorFlow

This project demonstrates how to perform image classification using a pre-trained ResNet50 model in TensorFlow/Keras, and log the results, model architecture, and visualizations to TensorBoard.  It also includes a directory monitoring feature to automatically process new images added to a specified input directory.  A Raspberry Pi camera integration is included to capture images directly into the processing pipeline.

## Features

* **Automatic Image Processing:** Monitors a designated input directory for new images and processes them automatically.
* **TensorBoard Integration:** Logs model architecture, model visualization, classified images, and predictions to TensorBoard for easy monitoring and analysis.
* **Image Annotation:** Adds predicted class and confidence score as a label to the processed images.
* **Directory Management:** Moves processed images from the input directory to a processed directory.
* **Error Handling:** Includes basic error handling for file processing issues.
* **Live Camera Integration:** Includes a command to capture images from a Raspberry Pi camera using `libcamera-still` and place them in the input directory.

## Requirements

* Python 3
* TensorFlow/Keras
* OpenCV (cv2)
* NumPy
* `libcamera-still` (for Raspberry Pi camera capture)

```bash
pip install tensorflow opencv-python numpy
Setup
Clone the repository:
Bash

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git) # Replace with your repo URL
cd YOUR_REPOSITORY_NAME
Create the necessary directories:
Bash

mkdir -p img/input img/processed
(Optional) Install libcamera-still if you intend to use the Raspberry Pi camera capture feature. Refer to the documentation for your Raspberry Pi OS for installation instructions.
Usage
Place images you want to classify in the img/input directory.  Alternatively, use the Raspberry Pi camera integration (see below).

Run the Python script (replace your_script_name.py with the actual name):

Bash

python your_script_name.py
Start TensorBoard to view the logs:
Bash

tensorboard --logdir logs/fit
Open TensorBoard in your browser (usually at http://localhost:6006).

The script will continuously monitor the img/input directory.  Processed images will be moved to the img/processed directory.

Raspberry Pi Camera Integration
The script includes a command to capture images from a Raspberry Pi camera using libcamera-still and save them to the img/input directory. This command is executed every 2 seconds within the monitor_directory function. You can adjust the camera settings (width, height, output file name) as needed in the script.  The default command used is:

Bash

libcamera-still --width 1536 --height 1024 -o img/input/smaller.jpg
Important: Ensure libcamera-still is installed and configured correctly on your Raspberry Pi.

Project Structure
YOUR_REPOSITORY_NAME/
├── img/
│   ├── input/        # Input images for classification
│   └── processed/   # Processed images
├── logs/             # TensorBoard logs
│   └── fit/
├── your_script_name.py # The main Python script
└── README.md         # This file
Code Explanation (Key Parts)
The Python script performs the following key actions:

Loads Pre-trained Model: Loads the ResNet50 model with pre-trained weights from ImageNet.
Logs Model Information: Logs the model architecture summary and a visualization of the model to TensorBoard.
classify_and_log_image() Function:
Loads and preprocesses the input image.
Performs prediction using the ResNet50 model.
Decodes the predictions to get the class name and confidence score.
Annotates the image with the prediction information.
Logs the annotated image and prediction text to TensorBoard.
monitor_directory() Function:
Continuously monitors the img/input directory for new files.
For each new file, calls classify_and_log_image() to process it.
Moves the processed image to the img/processed directory.
Includes the libcamera-still command to capture images from the camera.
Main Loop: Starts the monitor_directory() function to begin the image processing loop.
Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
[Choose a license - e.g., MIT License]
