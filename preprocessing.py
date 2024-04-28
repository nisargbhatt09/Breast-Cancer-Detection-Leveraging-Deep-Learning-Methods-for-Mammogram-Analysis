import os
import pydicom
from PIL import Image
import numpy as np

classes = ['cancer', 'non_cancer']


def dcm_to_png(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(source_folder):
        if file.endswith('.dcm'):
            dicom_image = pydicom.dcmread(os.path.join(source_folder, file))
            pil_image = Image.fromarray(dicom_image.pixel_array)
            pil_image.save(os.path.join(output_folder, file.replace('.dcm', '.png')))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def windowing():
    png_dir = os.path.expanduser('/kaggle/input/rsna-png-dataset/kaggle/working/cancer/')
    
    window_level = 127
    window_width = 255
    
    png_files = glob.glob(os.path.join(png_dir, '*.png'))

    for file in png_files:
    # Read the PNG file
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    # Perform windowing
    min_window = window_level - (window_width / 2)
    max_window = window_level + (window_width / 2)
    img_windowed = np.clip(img, min_window, max_window)
    img_windowed = (img_windowed - min_window) / window_width * 255
    
    # Save the windowed image as a new PNG file
    cv2.imwrite(f'{os.path.splitext(file)[0]}_windowed.png', img_windowed)

def hist_eq(src_dir, tar_dir):
    # png_dir = os.path.expanduser('/kaggle/input/rsna-png-dataset/kaggle/working/cancer/')
    # enhanced_dir = os.path.expanduser('/kaggle/working/enhanced_cancer_img')
    png_files = glob.glob(os.path.join(png_dir, '*.png'))

    for file in png_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        
        img_eq = cv2.equalizeHist(img)
        
        min_val = img_eq.min()
        max_val = img_eq.max()
        img_stretched = (img_eq - min_val) / (max_val - min_val) * 255
        
        filename = os.path.basename(file)
        cv2.imwrite(os.path.join(enhanced_dir, f'{os.path.splitext(filename)[0]}_enhanced.png'), img_stretched)

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

image_files = []
labels = []

for class_name in classes:
    class_dir = os.path.join(png_dir, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.png'):
            image_files.append(os.path.join(class_dir, file_name))
            labels.append(classes.index(class_name))

train_files, val_files, train_labels, val_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42)

train_dataset = MammogramDataset(train_files, train_labels, transform)
val_dataset = MammogramDataset(val_files, val_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(classes))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    
dcm_to_png('dicom_folder', 'png_folder')
