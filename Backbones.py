import torch
import torch.nn as nn
from torchvision import models, transforms


class SimpleCNN:
    """CNN classifier that makes predictions on numbers (categories) of the MNIST dataset."""

    def __init__(self):
        self.num_outputs = 10
        self.net = nn.Sequential(  # we are populating the net attribute via creation of a sequential container
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # we are moving from 28x28 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # we are moving from 14x14 -> 6x6
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 2048),  # we are moving from 6x6 -> 2x2
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, self.num_outputs))  # self.num_outputs are 10 classes in this case

        self.preprocess_train = transforms.Compose([transforms.RandomRotation(45),
                                                    # transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor()])

        self.preprocess_eval = transforms.Compose([transforms.ToTensor()])


class MLP1:
    """1-hidden layer MLP classifier that makes predictions on numbers (categories) of the MNIST dataset."""

    def __init__(self):

        self.num_outputs = 10
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 100),
            nn.Sigmoid(),
            torch.nn.Linear(100, self.num_outputs))

        self.preprocess_train = transforms.Compose([transforms.RandomRotation(45), transforms.ToTensor()])
        self.preprocess_eval = transforms.Compose([transforms.ToTensor()])


class MLP2:
    """2-hidden layer MLP classifier that makes predictions on numbers (categories) of the MNIST dataset."""

    def __init__(self):

        self.num_outputs = 10
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            torch.nn.Linear(100, self.num_outputs))

        self.preprocess_train = transforms.Compose([transforms.RandomRotation(45), transforms.ToTensor()])
        self.preprocess_eval = transforms.Compose([transforms.ToTensor()])













