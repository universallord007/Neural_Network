import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import CatsVsDogs
import warnings
from PIL import Image
import torch.nn.functional as F

warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder("PetImages", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = CatsVsDogs()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# torch.save(model.state_dict(), "catdog_model.pth")

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    classes = ["Cat", "Dog"]
    print("Prediction:", classes[predicted.item()])
    print("Confidence:", round(probabilities[0][predicted.item()].item() * 100, 2), "%")

predict("Dog_ML.jpg")