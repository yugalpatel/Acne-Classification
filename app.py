from model import load_model, transform_image, predict_acne_type, get_recommendations
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
import torch

app = Flask(__name__)
model = load_model()  # Load the trained model at the start

# Root route to display a simple message
@app.route('/')
def home():
    return "Welcome to the Acne Type Detector API. Use the /api/analyze endpoint to upload images."

# Endpoint for analyzing the image
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image_tensor = transform_image(io.BytesIO(image_file.read()))  # Transform the image
    acne_type = predict_acne_type(model, image_tensor)  # Get the acne type prediction
    recommendations = get_recommendations(acne_type)

    return jsonify({'acneType': acne_type, 'recommendations': recommendations})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the port from the environment or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)



