from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

NetIO = Union[FloatTensor, Variable]

# import numpy as np
from IPython import embed


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module, set_size, img_shape):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.set_size = set_size
        self.img_shape = img_shape

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        # embed()
        # exit()

        x = torch.reshape(x, (-1, *self.img_shape))

        x = self.phi.forward(x)

        x = torch.reshape(x, (-1, self.set_size, x.shape[1]))

        # x = x.unsqueeze(1)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.

        # x = torch.sum(x, dim=0, keepdim=True)
        x = torch.sum(x, dim=1, keepdim=True)

        # embed()

        # x = torch.squeeze(x, 1)
        x = torch.reshape(x, (-1, x.shape[2]))

        # compute the output
        out = self.rho.forward(x)

        return out


class SmallMNISTCNNPhi(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc1_drop = nn.Dropout2d()
    #     self.fc2 = nn.Linear(50, 10)

    # def forward(self, x: NetIO) -> NetIO:
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = self.conv2_drop(self.conv2(x))
    #     x = F.relu(F.max_pool2d(x, 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc1_drop(x)
    #     x = F.relu(self.fc2(x))
    #     return x
    def __init__(self, dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 100),
            nn.ELU(inplace=True),
            nn.Linear(100, 36),
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        out = self.model(img)
        return out


class SmallRho(nn.Module):
    def __init__(self, input_size: int, output_size: int=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(36, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, self.output_size),
        )

    def forward(self, x: NetIO) -> NetIO:
        x = self.model(x)
        return x
