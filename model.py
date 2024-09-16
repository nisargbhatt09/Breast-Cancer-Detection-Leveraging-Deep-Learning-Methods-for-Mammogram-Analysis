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

for epoch in range(10):  # loop over the dataset multiple times
    for inputs, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
      
