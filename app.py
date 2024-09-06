from model import load_model, transform_image, predict_acne_type, get_recommendations
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch

app = Flask(__name__)
model = load_model()  # Load the trained model at the start

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
    app.run(debug=True, port=5000)


