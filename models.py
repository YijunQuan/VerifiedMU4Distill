import torch.nn as nn
import torch.nn.functional as F


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output: 28x28x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Output: 28x28x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half (14x14x64)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # First FC layer
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes for MNIST)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        x = self.fc2(x)  # Output layer
        return x

class CNN_SVHN(nn.Module):
    def __init__(self):
        super(CNN_SVHN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Output: 32x32x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half (16x16x64)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # First FC layer
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes for MNIST)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        x = self.batch_norm(x)
        x = self.fc2(x)  # Output layer
        return x