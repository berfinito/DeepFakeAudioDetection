import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # (128x96) -> (128x96)
        self.pool = nn.MaxPool2d(2, 2)  # (128x96) -> (64x48)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(24*32*64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (128x96) -> (64x48)
        x = self.pool2(F.relu(self.conv2(x))) # (64x48) -> (32x24)


        x = x.view(-1, 24*32*64)

        x = self.fc2(F.relu(self.fc1(x)))

        return x
