
from image_transformation import transformations
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import torch

from model_training import train_model
from net import Net
import config

train_set = torchvision.datasets.CIFAR10(root=config.data_path, train=True, transform=transformations()['train'],
                                         download=True)
val_set = torchvision.datasets.CIFAR10(root=config.data_path, train=False, transform=transformations()['val'])


datasets = {'train':train_set, 'val':val_set}
dataloaders = {x: torch.utils.data.DataLoader(dataset=datasets[x], batch_size=config.batch_size, shuffle=True)
              for x in ['train', 'val']}

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)


model = Net().to(config.device)
# x = torch.randn(512, 3, 32, 32).to(device)
# # model(x)
# # print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=.9)
_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
model, losses = train_model(model,
                       criterion,
                       optimizer,
                       _lr_scheduler,
                       num_epochs=5)