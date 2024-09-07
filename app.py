import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model import build_model, get_recommendations
from celery_config import make_celery

# Initialize Flask and Celery
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)
celery = make_celery(app)

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

# Define the Celery task for image analysis
@celery.task(name="app.analyze_image_task")
def analyze_image_task(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = preprocess_image(image)

    with torch.no_grad():
        predictions = model(image)
        acne_type_index = predictions.argmax(dim=1).item()
        acne_types = ['acne_comedonica', 'acne_conglobata', 'acne_papulopistulosa']
        acne_type = acne_types[acne_type_index]
        recommendations = get_recommendations(acne_type)

    return {'acneType': acne_type, 'recommendations': recommendations}

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    # Queue the task in Celery
    task = analyze_image_task.delay(image_data)

    return jsonify({'task_id': task.id})

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    task = analyze_image_task.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)  # Exception message
        }
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(debug=True, host='0.0.0.0', port=port)

