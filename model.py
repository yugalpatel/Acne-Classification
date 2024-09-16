import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score
from PIL import Image

# Define dataset paths
train_dir = '/Users/yugalpatel/Downloads/acne/Dataset/train'
test_dir = '/Users/yugalpatel/Downloads/acne/Dataset/test'

# Define hyperparameters
batch_size = 16
num_epochs = 25
learning_rate = 0.001

# Define data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, shear=15),
        transforms.ColorJitter(brightness=0.25),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),  # Convert to tensor before applying RandomErasing
        transforms.RandomErasing(p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create a custom dataset class
class AcneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['acne_comedonica', 'acne_conglobata', 'acne_papulopustulosa']
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create datasets
train_dataset = AcneDataset(root_dir=train_dir, transform=data_transforms['train'])
test_dataset = AcneDataset(root_dir=test_dir, transform=data_transforms['test'])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Function to build the model
def build_model():
    model = models.resnet50(pretrained=True)
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

    return model

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
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


