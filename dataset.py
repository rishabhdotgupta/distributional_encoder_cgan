from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from matplotlib import pyplot as plt

# from .settings import DATA_ROOT

from IPython import embed

MNIST_TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


class MNISTSummation(Dataset):
    def __init__(self, set_size: int, train: bool = True, transform: Compose = None):

        self.set_size = set_size
        self.train = train
        self.transform = transform

        self.mnist = MNIST("../../data/mnist", train=self.train,
                           transform=self.transform, download=True)
        mnist_len = self.mnist.__len__()
        mnist_items_range = np.arange(0, mnist_len)

        self.mnist_items = []

        for l in range(10):
            ids = self.mnist.targets == l
            for i in range(int(len(mnist_items_range[ids]) / set_size)):
                self.mnist_items.append(
                    mnist_items_range[ids][i * set_size:(i + 1) * set_size])

    def __len__(self) -> int:
        # return self.dataset_len
        return len(self.mnist_items)

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        # print(item)
        # embed()
        mnist_items = self.mnist_items[item]
        images1 = []
        for mi in mnist_items:
            img, target = self.mnist.__getitem__(mi)
            images1.append(img)

        return torch.stack(images1, dim=0), torch.LongTensor([target])


class Parametric(Dataset):
    def __init__(self, set_size: int, train: bool = True, transform: Compose = None):

        set_dist = []
        targets = []
        label = 0

        for i in range(100):
            x = torch.rand(set_size, 1)
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(35):
            m = torch.distributions.beta.Beta(torch.tensor([.5]), torch.tensor([.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(35):
            m = torch.distributions.beta.Beta(torch.tensor([.7]), torch.tensor([.3]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(40):
            m = torch.distributions.beta.Beta(torch.tensor([.2]), torch.tensor([.7]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(50):
            m = torch.distributions.exponential.Exponential(torch.tensor([1.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(50):
            m = torch.distributions.exponential.Exponential(torch.tensor([2.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(100):
            m = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(40):
            m = torch.distributions.laplace.Laplace(torch.tensor([1.0]), torch.tensor([1.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(59):
            m = torch.distributions.laplace.Laplace(torch.tensor([.5]), torch.tensor([1.0]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(100):
            m = torch.distributions.log_normal.LogNormal(torch.tensor([0.0]), torch.tensor([0.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(59):
            m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(50):
            m = torch.distributions.normal.Normal(torch.tensor([0.3]), torch.tensor([0.5]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        for i in range(100):
            m = torch.distributions.studentT.StudentT(torch.tensor([2.0]))
            x = m.sample([set_size])
            set_dist.append(x)
            targets.append(torch.LongTensor([label]))
        label += 1

        # for i in range(100):
        #     # m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.25]))
        #     m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        #     x = m.sample([set_size])
        #     set_dist.append(x)
        #     targets.append(torch.LongTensor([label]))
        # label += 1

        # for i in range(100):
        #     mix = torch.distributions.Categorical(torch.ones(2,))
        #     # comp = torch.distributions.Normal(torch.tensor([-0.3, 0.3]), torch.tensor([0.15, 0.15]))
        #     comp = torch.distributions.Normal(torch.tensor([-2.0, 2.0]), torch.tensor([1.0, 1.0]))
        #     gmm = torch.distributions.MixtureSameFamily(mix, comp)
        #     x = gmm.sample([set_size])
        #     set_dist.append(x.unsqueeze(1))
        #     targets.append(torch.LongTensor([label]))
        # label += 1

        # for i in range(100):
        #     m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        #     x = m.sample([set_size])
        #     set_dist.append(x)
        #     targets.append(torch.LongTensor([label]))
        # label += 1

        # for i in range(100):
        #     m = torch.distributions.normal.Normal(torch.tensor([-1.0]), torch.tensor([1.0]))
        #     x = m.sample([set_size])
        #     set_dist.append(x)
        #     targets.append(torch.LongTensor([label]))
        # label += 1

        # plt.figure()
        # # plt.subplot(211)
        # plt.hist(set_dist[99].cpu().numpy(), bins=50)
        # # plt.subplot(212)
        # plt.hist(set_dist[100].detach().cpu().numpy(), bins=50)
        # plt.savefig("gen_param/real.png")

        self.data = torch.stack(set_dist).float()
        self.targets = torch.stack(targets)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[FloatTensor, FloatTensor]:

        x = self.data[index]
        target = self.targets[index]
        return x, target
