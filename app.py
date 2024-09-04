import time
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model import build_model, get_recommendations

app = Flask(__name__)

# Load the model structure
model = build_model()
device = torch.device("cpu")  # Force the model to run on the CPU
model = model.to(device)

# Load the trained weights
model.load_state_dict(torch.load('acne_model.pth', map_location=device))
model.eval()  # Set model to evaluation mode

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Reduce image size to 256x256
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    start_time = time.time()  # Start timing
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    print(f"Image received: {image_file.filename}")
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        print(f"Image loaded in {time.time() - start_time:.2f} seconds")
        
        image = preprocess_image(image)
        print(f"Image preprocessed in {time.time() - start_time:.2f} seconds")

        with torch.no_grad():
            predictions = model(image)
            acne_type_index = predictions.argmax(dim=1).item()
            acne_types = ['acne_comedonica', 'acne_conglobata', 'acne_papulopistulosa']
            acne_type = acne_types[acne_type_index]
            recommendations = get_recommendations(acne_type)

        print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        return jsonify({'acneType': acne_type, 'recommendations': recommendations})
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

