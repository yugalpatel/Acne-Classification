import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Function to build the model
def build_model():
    model = models.resnet50(pretrained=False)  # We're not using pre-trained weights from ImageNet
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3 classes for acne types
    return model

# Function to get recommendations based on acne type
def get_recommendations(acne_type):
    recommendations = {
        'acne_comedonica': ['Product A', 'Product B'],
        'acne_conglobata': ['Product C', 'Product D'],
        'acne_papulopistulosa': ['Product E', 'Product F']
    }
    return recommendations[acne_type]

# Load the trained model weights
def load_model():
    model = build_model()
    model.load_state_dict(torch.load('acne_model.pth', map_location=torch.device('cpu')))  # Load weights on CPU
    model.eval()  # Set the model to evaluation mode
    return model

# Transform the input image for model prediction
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict the acne type from the image
def predict_acne_type(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    acne_types = ['acne_comedonica', 'acne_conglobata', 'acne_papulopistulosa']
    return acne_types[predicted_class.item()]


