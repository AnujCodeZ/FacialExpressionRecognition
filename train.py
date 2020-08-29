import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

from model import FaceNet

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 64
lr = 3e-3
epochs = 20

# Data and preprocessing
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(48),
                                transforms.ToTensor()])

trainset = datasets.ImageFolder(root='data/train/', transform=transform)
testset = datasets.ImageFolder(root='data/test/', transform=transform)

trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=64,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=64,
                                          shuffle=True)

# Model
model = FaceNet()
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

# Training
for e in range(epochs):
    batch_num = 1
    for images, labels in trainloader:
        
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_num % 100 == 0:
            print(f'Batch: {batch_num}, Loss: {loss.item()}')
        batch_num += 1
    
    print(f'Epoch: {e+1}, Loss: {loss.item()}')

# Validate
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f'Test accuracy: {(correct / total)*100}%')

# Saving model
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model_file = {
    'model_state': model.state_dict(),
    'label_names': label_names
}

torch.save(model_file, '.app/model.pth')