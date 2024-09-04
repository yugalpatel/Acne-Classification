import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Function to build the model
def build_model():
    model = models.resnet50(pretrained=False)  # pretrained=False since you're loading your trained weights
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

# Load pre-trained ResNet model
def load_model():
    model = build_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load('acne_model.pth', map_location=device))  # Load trained model weights
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess image for model input
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict the acne type
def predict_acne_type(image, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

