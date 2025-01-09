# Import essential libraries
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import pydicom
from captum.attr import IntegratedGradients, GuidedGradCAM, Occlusion
import matplotlib.pyplot as plt

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing
df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')
df_test = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')

# Basic transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset class
class MammogramDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Data preparation
image_files = []
labels = []
classes = ['cancer', 'non_cancer']

for class_name in classes:
    class_dir = os.path.join(png_dir, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.png'):
            image_files.append(os.path.join(class_dir, file_name))
            labels.append(classes.index(class_name))

# Split data and create dataloaders
train_files, val_files, train_labels, val_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42)
train_dataset = MammogramDataset(train_files, train_labels, transform)
val_dataset = MammogramDataset(val_files, val_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup and training
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(classes)).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        print(f'Epoch {epoch+1}')
        print(classification_report(y_true, y_pred, target_names=classes))
    model.train()


#Explainable AI
def visualize_model_explanations(model, image, label, class_names):
    # Initialize explainable AI methods
    integrated_gradients = IntegratedGradients(model)
    guided_gradcam = GuidedGradCAM(model, model.blocks[-1])
    occlusion = Occlusion(model)
    
    # Get attributions
    ig_attr = integrated_gradients.attribute(image, target=label)
    gradcam_attr = guided_gradcam.attribute(image, target=label)
    occlusion_attr = occlusion.attribute(image,
                                       target=label,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(3, 15, 15))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image.squeeze().permute(1,2,0))
    plt.title('Original Image')
    
    plt.subplot(1, 4, 2)
    plt.imshow(ig_attr.squeeze().sum(0).detach().cpu().numpy())
    plt.title('Integrated Gradients')
    
    plt.subplot(1, 4, 3)
    plt.imshow(gradcam_attr.squeeze().sum(0).detach().cpu().numpy())
    plt.title('Guided GradCAM')
    
    plt.subplot(1, 4, 4)
    plt.imshow(occlusion_attr.squeeze().sum(0).detach().cpu().numpy())
    plt.title('Occlusion')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Select a sample image from validation set
sample_image, sample_label = next(iter(val_loader))
sample_image = sample_image[0].unsqueeze(0).to(DEVICE)
sample_label = sample_label[0].to(DEVICE)

# Generate explanations
visualize_model_explanations(model, sample_image, sample_label, classes)
