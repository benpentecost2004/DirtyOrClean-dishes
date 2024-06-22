import pandas as pd
import os
from PIL import Image
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Change the directory path to match your own directory
for dirname, _, filenames in os.walk('/Users/benpentecost/Documents/CodingProjects/PythonProjects/DirtyOrCleanDishes'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Change directories
train_dir = '/Users/benpentecost/Documents/CodingProjects/PythonProjects/DirtyOrCleanDishes/archive/train'
test_dir = '/Users/benpentecost/Documents/CodingProjects/PythonProjects/DirtyOrCleanDishes/archive/test'

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

epochs = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logps, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs}..")
    print(f"Training loss: {running_loss / len(train_loader)}")
    print(f"Training accuracy: {accuracy:.4f}")

class TestDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

test_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith(('png', 'jpg', 'jpeg'))]
test_dataset = TestDataset(test_files, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
results = []

with torch.no_grad():
    for inputs, paths in test_loader:
        inputs = inputs.to(device)
        logps = model(inputs)
        ps = torch.exp(logps)
        _, preds = torch.max(ps, 1)
        for path, pred in zip(paths, preds):
            label = 'cleaned' if pred.item() == 0 else 'dirty'
            results.append({'file': os.path.basename(path), 'label': label})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('test_predictions.csv', index=False)

print("test_predictions:")
print(results_df)