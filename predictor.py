import torch
import numpy as np
from torchvision import transforms
from model import FaceNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_file = torch.load('model.pth', map_location=device)
model_state = model_file['model_state']
label_names = model_file['label_names']

model = FaceNet()
model.load_state_dict(model_state)
model.to(device)
model.eval()

transform = transforms.ToTensor()

def predict(img):

    img = transform(img)
    img = img.to(device)
    img = img.view(1, 1, 48, 48)
    out = model(img)
    _, pred = torch.max(out, 1)
    
    return label_names[pred.item()]