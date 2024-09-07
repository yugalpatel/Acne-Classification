import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model import build_model, get_recommendations

app = Flask(__name__)

# Load the model structure
model = build_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the trained weights
model.load_state_dict(torch.load('acne_model.pth', map_location=device))
model.eval() 

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0) 
    return image.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image = preprocess_image(image)

    with torch.no_grad():
        predictions = model(image)
        acne_type_index = predictions.argmax(dim=1).item()
        acne_types = ['acne_comedonica', 'acne_conglobata', 'acne_papulopistulosa']
        acne_type = acne_types[acne_type_index]
        recommendations = get_recommendations(acne_type)

    return jsonify({'acneType': acne_type, 'recommendations': recommendations})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(debug=True, host='0.0.0.0', port=port)
