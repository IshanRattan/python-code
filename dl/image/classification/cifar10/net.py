

from torchvision import models
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(Net, self).__init__()

        # Load pre-trained VGG model (e.g., VGG16)
        vgg = models.vgg16(pretrained=True)

        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=2),
            *list(vgg.features.children())[1:-1]
        )


        for param in self.features.parameters():
            param.requires_grad = True

        # Custom dense layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 2048),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x