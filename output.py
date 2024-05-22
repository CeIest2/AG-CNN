import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the Classifier model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 25)
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the CSVDataset class
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=1)
        self.labels = self.data.iloc[:, 0]
        self.pixels = self.data.iloc[:, 1:].values.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        pixels = torch.tensor(self.pixels[index]).reshape((1, 28, 28))
        return pixels, label

# Load the trained model and map it to the CPU
model = Classifier()
model.load_state_dict(torch.load('model_checkpoint.pth', map_location=torch.device('cpu')))

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the test dataset from the CSV file
test_dataset = CSVDataset('dataset/sign_mnist_test/sign_mnist_test.csv')

# Create a DataLoader to load the test data in batches
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Disable the model's training mode
model.eval()

# Initialize counters for correct predictions and total predictions
num_correct = 0
num_total = 0
examples = 5

# Mapping of class numbers to letters
class_map = {i: chr(65 + i) for i in range(25)}

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 7))

# Disable gradient computation for inference
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        if idx >= examples:
            break
        pixels, labels = data
        pixels, labels = pixels.to(device), labels.to(device)
        outputs = model(pixels)
        _, predicted = torch.max(outputs.data, 1)

        # Plot the image in the subplot
        row, col = divmod(idx, 3)
        ax = axs[row, col]
        ax.imshow(pixels.cpu().squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {predicted.item()} ({class_map[predicted.item()]}), Actual: {labels.item()} ({class_map[labels.item()]})')
        ax.axis('off')

        # Update the counters
        num_correct += (predicted == labels).sum().item()
        num_total += labels.size(0)

# Hide any unused subplots
for i in range(examples, 6):
    row, col = divmod(i, 3)
    fig.delaxes(axs[row, col])

# Show the plots
plt.tight_layout()
plt.show()

# Calculate the model's accuracy
accuracy = num_correct / num_total
print(f'Model Accuracy: {accuracy:.4f}')
