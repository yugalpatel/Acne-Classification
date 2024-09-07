import os
import io
import logging
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import build_model, get_recommendations

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model structure
model = build_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the trained weights
logger.info("Loading model weights...")
model.load_state_dict(torch.load('acne_model.pth', map_location=device))
model.eval()

# Image preprocessing function
def preprocess_image(image):
    logger.info("Preprocessing image...")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

@app.route('/')
def home():
    logger.info("Rendering home page...")
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    logger.info("Image analysis request received...")
    
    if 'image' not in request.files:
        logger.error("No image provided in request.")
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read and preprocess the image
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = preprocess_image(image)

        # Perform the model prediction
        logger.info("Running model inference...")
        with torch.no_grad():
            predictions = model(image)
            acne_type_index = predictions.argmax(dim=1).item()

        # Map the prediction to an acne type
        acne_types = ['acne_comedonica', 'acne_conglobata', 'acne_papulopistulosa']
        acne_type = acne_types[acne_type_index]
        logger.info(f"Predicted acne type: {acne_type}")

        # Get product recommendations
        recommendations = get_recommendations(acne_type)
        logger.info(f"Recommendations: {recommendations}")

        return jsonify({'acneType': acne_type, 'recommendations': recommendations})
    
    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    logger.info(f"Starting app on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port)

