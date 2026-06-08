import torch
import torch.nn as nn
from torchvision import transforms , datasets , models
from torch.utils.data import DataLoader

# We transformed the image size for resnet18 cuz resnet can only understand the images which are (224x224)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# Here we have loaded the images from the Data folder to the dataset variable
dataset = datasets.ImageFolder(
    root = "Data",
    transform = transform
)

# Here we have specified the batch size as well as allowed shuffling
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
)
# Here we are using the resnet model for the transfer learning thing
model = models.resnet18(weights="DEFAULT")

# Here we have freezed everything for feature extraction
for params in model.parameters():
    params.requires_grad = False


# We have like many classes in the resnet but we only need 2 of them so we are changinng the final classifier from that many to only 2 cuz we only need to identify the two things which are cats and dogs
model.fc = nn.Linear(
    model.fc.in_features , 2
)
# Declaring the loss function as well as the optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(),lr = 0.001)


# Running the training loop
for epoch in range(10):
    for images, labels in dataloader:
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Here we are checking out the accuracy of our model based on its previous training 
correct = 0 
total = 0 
with torch.no_grad():
    for images, labels in dataloader:
        output = model(images)
        _, prediction = torch.max(output,1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    print(f"Accuracy is :{100*correct/total:.2f}% ")
