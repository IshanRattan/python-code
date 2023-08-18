
from torchvision import transforms, datasets
from configs.config import data_dir

import os


def transformations():
    # Image transformations
    data_transformations = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transformations

def create_datasets(data_dir, transformation):
    # Create image folders for our training and validation data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              transformation[x])
                      for x in ['train', 'val']}
    return image_datasets

def get_dataset_size(dataset):
    # Obtain dataset sizes from image_datasets
    return {x: len(dataset[x]) for x in ['train', 'val']}

def train_model():
    transformation = transformations()
    datasets = create_datasets(data_dir, transformation)
    dataset_sizes = get_dataset_size(datasets)
    # Use image_datasets to sample from the dataset
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                                  shuffle=True)
                   for x in ['train', 'val']}
