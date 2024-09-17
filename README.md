# Acne Type Classification using Deep Learning

## Overview

This project uses a ResNet50 deep learning model to classify different types of acne based on images. The model predicts one of the three acne types: **Acne Comedonica**, **Acne Conglobata**, or **Acne Papulopustulosa**. The solution focuses on leveraging image classification techniques to assist in identifying acne types from user-uploaded images.

This project was developed with the intention of showcasing a practical application of computer vision in the dermatology space. The trained model is integrated into a web interface where users can upload an image, and the model will return the predicted acne type along with ingredient-based recommendations for potential treatments. This project is part of an effort to apply machine learning to real-world health and skincare problems.

### Motivation

The idea behind this project is to utilize cutting-edge deep learning techniques to help classify acne types. It serves as a starting point for implementing more advanced applications in computational dermatology. By combining convolutional neural networks (CNN) with Flask, the project demonstrates a real-world, end-to-end machine learning solution.

## Project Features

- **Image Classification**: Classifies three distinct acne types.
- **Ingredient Recommendation**: Provides ingredient recommendations based on acne type, including both over-the-counter and prescription ingredients.
- **Pretrained Model**: Uses a pretrained ResNet50 architecture for efficient image classification.
- **Web Interface**: A simple web interface for users to upload an image and get the classification and recommendations.

## Model Architecture

The model is based on the **ResNet50** architecture, a deep convolutional neural network with 50 layers that utilizes residual learning to address the vanishing gradient problem, enabling the training of very deep networks. ResNet50's architecture consists of a series of convolutional layers, batch normalization, and ReLU activation functions, followed by identity shortcut connections that skip layers, allowing the network to learn more complex features without degradation. In this project, the pre-trained ImageNet weights were replaced, and the fully connected layer was modified to output three classes corresponding to the acne types. This allows the model to capture intricate skin texture patterns while leveraging transfer learning to reduce the training time and improve generalization on the relatively small dataset. The final classification layer uses a softmax activation function to provide probabilistic predictions for each acne type.

## Dataset

The dataset used for training and testing was sourced from the [Acne Type Classification Dataset](https://universe.roboflow.com/taschenbier/acne-type-classification). The dataset includes **754 images**, annotated by acne type in folder format.

### Dataset Preprocessing:

- **Auto-orientation of pixel data** (with EXIF-orientation stripping).
- Resize all images to **640x640** pixels.

### Augmentation Applied:

Each image has been augmented to increase the dataset size and improve model robustness. The following augmentations were applied:
- **50% probability of horizontal flip**
- **Random rotation** between -15 and +15 degrees
- **Random shear** between -15° to +15° horizontally and -15° to +15° vertically
- **Random brightness adjustment** between -25% and +25%
- **Random Gaussian blur** of between 0 and 0.25 pixels
- **Salt and pepper noise** applied to 5% of the pixels

## Results

The model achieved an accuracy of **80%** on the validation set. This accuracy reflects the model’s ability to differentiate between the three types of acne, showing potential for real-world application in dermatology.

## Instructions to Run Locally

### Prerequisites

Ensure that you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- [Redis](https://redis.io/) (for Celery task queue)

### Clone the Repository

```bash
git clone https://github.com/yugalpatel/Acne-Classification.git
cd Acne-Classification
```

### Install Dependencies

Run the following command to install all necessary dependencies:

```bash
pip install -r requirements.txt
```

### Setting Up Redis

To run the project with Celery for background task processing, you'll need Redis running on your system:

1. Install Redis on your machine by following the official [Redis installation guide](https://redis.io/docs/getting-started/installation/).
2. Start Redis by running:

   ```bash
   redis-server
   ```

### Running the Project

1. **Start the Celery Worker**:

   Open a new terminal window and run:

   ```bash
   celery -A app.celery worker --loglevel=info
   ```

2. **Run the Flask App**:

   In another terminal window, start the Flask application:

   ```bash
   python app.py
   ```

3. **Open in Browser**:

   Open your web browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```

4. **Upload an Image**:

   Use the web interface to upload an image, and the model will classify the acne type and recommend appropriate ingredients.

### Note

If running on a remote server, make sure the necessary ports are open, and update the Flask app configuration accordingly.

## Technologies Used

- **Python**: The core programming language for the project.
- **PyTorch**: For building and training the deep learning model.
- **Torchvision**: For image transformations and model architecture.
- **Flask**: A lightweight framework for the web interface.
- **Celery**: For background task management (image processing).
- **Redis**: For managing Celery tasks.
- **Gunicorn**: For running the Flask application in production.

## Future Work

- **Mobile Application Integration**: Implement this system in a mobile app for easier user access.
- **Expand Dataset**: Increase the dataset size to improve accuracy.
- **More Skin Conditions**: Extend the model to classify other skin conditions (e.g., eczema, rosacea).
- **User Personalization**: Include a feature for users to receive personalized treatment plans.

## Acknowledgments

This project was made possible by the **[Acne Type Classification Dataset](https://universe.roboflow.com/taschenbier/acne-type-classification)** from Roboflow.

- **Dataset Details**: The dataset includes 754 images, resized to 640x640, with the following augmentations applied to each image:
  - Horizontal flip
  - Random rotation (-15 to +15 degrees)
  - Shear (-15° to +15°)
  - Brightness adjustment (-25% to +25%)
  - Gaussian blur (0 to 0.25 pixels)
  - Salt and pepper noise (5% of pixels)

Special thanks to the contributors of this dataset and the PyTorch community for supporting this project.
