import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image

# Load data
my_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomGrayscale(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # Improves the training. Los valores están cogidos al azar y como tal no hacen nada.
])

dataset = datasets.ImageFolder('ImageSet', transform=my_transforms) # ImageFolder espera la ruta que hay antes de las carpetas con las clases


num_img = 1
for _ in range(5):
    for img, label in dataset:
        save_image(img, 'GoofyModified' +str(num_img) + '.png')
        num_img += 1