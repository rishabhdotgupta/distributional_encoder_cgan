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
import sciplex
from pathlib import Path
import re

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

class SciplexData(Dataset):
    def __init__(self, set_size: int, num_sets: int, train: bool = True):

        # set_dist = []
        # targets = []
        # label = 0

        # for i in range(100):
        #     x = torch.rand(set_size, 1)
        #     set_dist.append(x)
        #     targets.append(torch.LongTensor([label]))
        # label += 1

        # self.data = torch.stack(set_dist).float()
        # self.targets = torch.stack(targets)

        data_path = Path('/home/yavuz/data/sciplex')
        sciplex2 = sciplex.SciPlex2(data_path / 'sciplex2', preprocess=True).dataset
        self.conditions, set_values, self.treatments, self.doses = self.extract_sets_from_perturbations(sciplex2)
        self.data, self.targets, self.sets, self.labels = self.sinkhorn_distances_among_sets(set_values, self.conditions, sample_size=set_size, sample_count=num_sets, projection='pca')

    def extract_sets_from_perturbations(self, dataset, group_by:list=['perturbation_raw']):
        set_labels = []
        set_values = []
        set_treatments = []
        set_doses = []
        for group, indices in dataset.obs.groupby(group_by).indices.items():
            # embed()
            # exit()
            set_labels.append(group)
            # split = re.split("[\b\W\b]+", group)[1:-1]
            set_treatments.append(group.split(',')[0][2:-1])
            set_doses.append(group.split(',')[1][2:-2])
            # set_doses.append(float(group.split(',')[1][2:-2]))
            set_values.append(dataset[indices])
        return set_labels, set_values, set_treatments, set_doses

    def sinkhorn_distances_among_sets(self, sets, labels, sample_size=None, sample_count=None, batch_size=1, backend='geomloss', projection='umap'):
        if sample_size is not None and sample_count is not None:
            result_sets = []
            result_labels = []
            targets = []
            for set_idx, set in enumerate(sets):
                for i in range(sample_count):
                    random_idx = np.random.randint(len(sets), size=sample_size)
                    result_sets.append(set[random_idx])
                    result_labels.append(labels[set_idx])
                    # targets.append(torch.LongTensor([set_idx]))
                    targets.append(set_idx)
            sets = result_sets
            labels = result_labels

        data = [torch.from_numpy(set.obsm[f'X_{projection}']) for set in sets]
        return torch.stack(data), torch.LongTensor(targets), sets, labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[FloatTensor, FloatTensor]:

        x = self.data[index]
        target = self.targets[index]
        condition = self.labels[index]
        return x, target
