
from torchvision.transforms import transforms
from PIL import Image
import torch

import config

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def predict(img_path, model):
    data = load_image(img_path).to(config.device)
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    return config.class_names[predicted.item()]


model = torch.load('model.pth')
img_path = "5.jpg"

print(predict(img_path, model))
