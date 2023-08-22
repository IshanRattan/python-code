
from torch.optim import lr_scheduler
from configs import config

import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import torch
import time
import copy
import os


def transformations():
    # Image transformations
    data_transformations = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transformations

def create_datasets(data_dir, transformation):
    # Create image folders for our training and validation data
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                              transformation[x])
                      for x in ['train', 'val']}
    return image_datasets

def get_dataset_size(dataset):
    # Obtain dataset sizes from image_datasets
    return {x: len(dataset[x]) for x in ['train', 'val']}

def get_device():
    # Change selected device to CUDA, a parallel processing platform, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def visualize_model(model, device, dataloaders, class_names, num_images=6):
    '''
    Function that will visualize results of the model
    '''
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp, title=None):
    """
    This function will make use of Matplotlib.pyplot's imshow() function for tensors.
    It will show the same number of images as the batch we defined.
    """
    # The transpose is required to get the images into the correct shape
    inp = inp.numpy().transpose((1, 2, 0))

    # Using default values for mean and std but can customize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # To visualize the correct colors
    inp = std * inp + mean

    # To view a clipped version of an image
    inp = np.clip(inp, 0, 1)

    # Visualize inp
    plt.imshow(inp)

    if title is not None:  # Plot title goes here
        plt.title(title)
    plt.pause(0.001)  # Enables the function to pause while the plots are updated

def pre_process():
    transformation = transformations()
    datasets = create_datasets(config.data_dir, transformation)
    dataset_sizes = get_dataset_size(datasets)

    # Obtain class_names from image_datasets
    class_names = datasets['train'].classes

    # Use image_datasets to sample from the dataset
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                                  shuffle=True)
                   for x in ['train', 'val']}

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # Plot the grid with a title that concatenates all the class labels
    imshow(out, title=[class_names[x] for x in classes])

    # Get a batch of test data
    inputs, classes = next(iter(dataloaders['val']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # Plot the grid with a title that concatenates all the class labels
    imshow(out, title=[class_names[x] for x in classes])
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    '''
    Function that will train model based on data provided.
    '''
    device = get_device()
    since = time.time()

    # Make a deep copy of the model provided
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data using the dataloader we defined
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass, tracking history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Computing loss statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Create a deep copy of the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()  # Print an empty line for nice formatting

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

def start_training():
    device = get_device()

    dataloaders, dataset_sizes, class_names = pre_process()

    # Load the resnet model
    model_ft = torchvision.models.resnet18(pretrained=True)

    # Obtaining the number of input features for our final layer
    num_ftrs = model_ft.fc.in_features

    # Since this is a binary classification task, we'll set the size of each output sample to 2. For multi-class classification, this can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Move the model to the device
    model_ft = model_ft.to(device)

    # We'll use CrossEntropyLoss(), which is a common loss function for classification problems
    criterion = nn.CrossEntropyLoss()

    # In this step, we'll optimize all parameters of the model
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=config.lr, momentum=config.momentum)

    # We'll decay learning rate (lr) by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.step_size, gamma=config.gamma)

    # Call our train_model() function with the ResNet model, the criterion, optimizer, learning rate scheduler, and number of epochs that we have defined.
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes,
                           num_epochs=config.num_epochs)

    visualize_model(model_ft, device, dataloaders, class_names)

    # Disable gradients for model_conv.parameters()
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    # Move the model to the device
    model_conv = model_conv.to(device)

    # Set criterion again
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as opposed to before
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=config.lr, momentum=config.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=config.step_size, gamma=config.gamma)

    # Train model_conv
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=config.num_epochs)

    # Visualize model
    visualize_model(model_conv, device, dataloaders, class_names)
    plt.show()

start_training()