
import torchvision
import torch

transform_test = torchvision.transforms.transforms.Compose([
    torchvision.transforms.transforms.transforms.ToTensor(),
    torchvision.transforms.transforms.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=False,
                                            transform=transform_test)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)