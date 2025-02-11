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

Loads Pre-trained Model
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
