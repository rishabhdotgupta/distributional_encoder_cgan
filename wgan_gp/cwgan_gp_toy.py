import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
# from torchvision import datasets
from sklearn import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from IPython import embed
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import ot

os.makedirs("images", exist_ok=True)
os.makedirs("dist", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=50, help="interval betwen image samples")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)
img_shape = (2,)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        # img_flat = img.view(img.shape[0], -1)
        validity = self.model(d_in)
        return validity


# Loss weight for gradient penalty
# lambda_gp = 10
lambda_gp = 0.1
lambda_ot = 0.0001

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# # Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

imgs, y = datasets.make_moons(n_samples=opt.batch_size, noise=0.05)
imgs = preprocessing.StandardScaler(with_std=False).fit_transform(imgs)
imgs = preprocessing.MaxAbsScaler().fit_transform(imgs)


def inf_train_gen():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        labels = []
        for i in range(opt.batch_size):
            point = np.random.randn(2) * .02
            # center = random.choice(centers)
            label = random.randint(0, 7)
            point[0] += centers[label][0]
            point[1] += centers[label][1]
            dataset.append(point)
            labels.append(label)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        labels = np.array(labels, dtype='int32')
        yield dataset, labels


data = inf_train_gen()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

# batches_done = 0
d_rg = []
w_rg = []
diff = []
for epoch in range(opt.n_epochs):
    # for i, (imgs, _) in enumerate(dataloader):

    # Configure input
    # real_imgs = Variable(imgs.type(Tensor))

    # # np.random.shuffle(imgs)
    perm = np.random.permutation(len(imgs))
    imgs = imgs[perm]
    y = y[perm]
    # imgs, y = data.__next__()

    batch_size = imgs.shape[0]

    real_imgs = Variable((torch.from_numpy(imgs).type(Tensor)))
    labels = Variable(torch.from_numpy(y).type(LongTensor))

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
    # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
    gen_labels = labels

    # Generate a batch of images
    fake_imgs = generator(z, gen_labels)

    # Real images
    real_validity = discriminator(real_imgs, labels)
    # Fake images
    fake_validity = discriminator(fake_imgs, gen_labels)
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels)
    # OT penalty
    ot_penalty = torch.mean(real_validity[y == 1]) - torch.mean(real_validity[y == 0])
    # Adversarial loss
    # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty + lambda_ot * ot_penalty

    d_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    # Train the generator every n_critic steps
    # if i % opt.n_critic == 0:
    if epoch % opt.n_critic == 0:

        # -----------------
        #  Train Generator
        # -----------------

        # Generate a batch of images
        fake_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        fake_validity = discriminator(fake_imgs, gen_labels)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        # )

        if epoch % opt.sample_interval == 0:

            # for c in range(opt.n_classes):
            size0 = real_validity[y == 0].shape[0]
            # size1 = fake_validity[y == 1].shape[0]
            size1 = real_validity[y == 1].shape[0]
            # val = torch.mean(real_validity[y == 0]) - torch.mean(fake_validity[y == 1])
            val = torch.mean(real_validity[y == 0]) - torch.mean(real_validity[y == 1])
            # r, g = np.ones((opt.batch_size,)) / opt.batch_size, np.ones((opt.batch_size,)) / opt.batch_size
            # M = ot.dist(real_imgs[y == 0].cpu().detach().numpy(), fake_imgs[y == 1].cpu().detach().numpy(), metric='euclidean')
            M = ot.dist(real_imgs[y == 0].cpu().detach().numpy(), real_imgs[y == 1].cpu().detach().numpy(), metric='euclidean')
            r, g = np.ones((size0,)) / size0, np.ones((size1,)) / size1
            wass = ot.emd2(r, g, M)
            d_rg.append(val.cpu().detach().numpy())
            w_rg.append(wass)
            diff.append(wass - val.cpu().detach().numpy())
            # break

            plt.figure()
            plt.plot(d_rg, label="Discriminator value function")
            plt.plot(w_rg, label="True Wasserstein Distance")
            plt.legend()
            plt.xlabel("Training Time")
            # plt.ylabel("Distances")
            plt.savefig("dist/%d.png" % epoch)

            plt.figure()
            plt.plot(diff, label="Difference")
            plt.legend()
            plt.xlabel("Training Time")
            # plt.ylabel("Distances")
            plt.savefig("dist/diff_%d.png" % epoch)

            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, d_loss.item(), g_loss.item())
            )

            # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            real_imgs = real_imgs.cpu().detach().numpy()
            plt.figure()
            plt.scatter(real_imgs[:, 0], real_imgs[:, 1], c=labels.cpu().detach().numpy())
            plt.savefig("images/real_%d.png" % epoch)
            plt.clf()
            fake_imgs = fake_imgs.cpu().detach().numpy()
            plt.figure()
            plt.scatter(fake_imgs[:, 0], fake_imgs[:, 1], c=labels.cpu().detach().numpy())
            plt.savefig("images/fake_%d.png" % epoch)
            plt.clf()

        # batches_done += opt.n_critic
