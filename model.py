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
        'Acne Comedonica': {
            'Description': 'Acne comedonica is characterized by whiteheads and blackheads, caused by blocked pores.',
            'Ingredients': [
                'Salicylic Acid - Helps to exfoliate the skin and unclog pores.',
                'Benzoyl Peroxide - Works as an antimicrobial agent, reducing bacteria and excess oil.',
                'Niacinamide - Reduces inflammation and improves skin texture.',
                'Retinoids (Prescription) - Promotes cell turnover to prevent clogged pores. *Consult a dermatologist for prescription use.*',
            ],
            'Disclaimer': 'Always consult with a dermatologist before using prescription ingredients.'
        },
        'Acne Conglobata': {
            'Description': 'Acne conglobata is a severe form of acne involving deep, painful cysts and inflammation.',
            'Ingredients': [
                'Benzoyl Peroxide - Reduces bacteria and inflammation.',
                'Azelaic Acid - Reduces redness and kills acne-causing bacteria.',
                'Alpha Hydroxy Acids (AHAs) - Helps to exfoliate the skin and reduce clogged pores.',
                'Isotretinoin (Prescription) - A powerful oral retinoid for severe acne. *Consult a dermatologist for prescription use.*',
            ],
            'Disclaimer': 'Always consult with a dermatologist before using prescription ingredients.'
        },
        'Acne Papulopustulosa': {
            'Description': 'Acne papulopustulosa involves inflamed pimples and pustules that are often painful.',
            'Ingredients': [
                'Benzoyl Peroxide - Reduces inflammation and kills acne-causing bacteria.',
                'Sulfur - Absorbs excess oil and has antibacterial properties.',
                'Clindamycin or Dapsone Gel (Prescription) - Helps with inflammation and bacterial growth. *Consult a dermatologist for prescription use.*',
            ],
            'Disclaimer': 'Always consult with a dermatologist before using prescription ingredients.'
        }
    }
    
    return recommendations.get(acne_type, {'error': 'Acne type not found'})
<<<<<<< HEAD
=======

# Load pre-trained ResNet model
model = build_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs-1}, Loss: {epoch_loss:.4f}')
>>>>>>> 66334eb4d6c045e89c081e583cdb86f980af42a0

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
<<<<<<< HEAD
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    acne_types = ['acne_comedonica', 'acne_conglobata', 'acne_papulopistulosa']
    return acne_types[predicted_class.item()]
=======
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Main function
if __name__ == "__main__":
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs)
    evaluate_model(trained_model, test_loader)
    # Save the trained model
    torch.save(trained_model.state_dict(), 'acne_model.pth')


>>>>>>> 66334eb4d6c045e89c081e583cdb86f980af42a0
