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
            nn.Linear(dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 10),
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

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.output_size)

    def forward(self, x: NetIO) -> NetIO:
        # embed()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x - xm)
        return x

class DeepSet_pc(nn.Module):

    def __init__(self, d_dim=256, x_dim=3, out_dim=10):
        super(DeepSet_pc, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim
        self.out_dim = out_dim

        self.phi = nn.Sequential(
            PermEqui1_max(self.x_dim, self.d_dim),
            nn.Tanh(),
            PermEqui1_max(self.d_dim, self.d_dim),
            nn.Tanh(),
            PermEqui1_max(self.d_dim, self.d_dim),
            nn.Tanh(),
        )

        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.d_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.out_dim),
            # nn.Linear(self.d_dim, self.d_dim),
        )
        # print(self)

    def forward(self, x):
        phi_output = self.phi(x)
        sum_output, _ = phi_output.max(1)
        ro_output = self.ro(sum_output)
        return ro_output
