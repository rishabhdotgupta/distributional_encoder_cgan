import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataset import MNISTSummation, Parametric
from encoder import InvariantModel, SmallMNISTCNNPhi, SmallRho
# from param_encoder import InvariantModel, SmallMNISTCNNPhi, SmallRho
from IPython import embed
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--enc_dim", type=int, default=10, help="output dim of the encoder")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--set_size", type=int, default=1, help="size of the sets")
opt = parser.parse_args()
print(opt)

os.makedirs("gen_images_%d_%d/" % (opt.batch_size, opt.set_size), exist_ok=True)
os.makedirs("real_images_%d_%d/" % (opt.batch_size, opt.set_size), exist_ok=True)
os.makedirs("tsne/", exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)
# img_shape = (opt.channels,)

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
            *block(opt.latent_dim + opt.enc_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        if labels.size(1) == 1:
            one_hot = torch.cuda.FloatTensor(labels.size(0), 10)
            labels = one_hot.scatter_(1, labels.data, 1)
            labels = labels.unsqueeze(1).expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
            # labels.squeeze(1)
            # labels = self.label_emb(labels)
            # labels = labels.expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
        else:
            # Concatenate label embedding and image to produce input
            # gen_input = torch.cat((self.label_emb(labels), noise), -1)
            # labels = torch.cat(int(noise.size(0) / labels.size(0)) * [labels])
            labels = labels.unsqueeze(1).expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
        gen_input = torch.cat((labels, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.enc_dim + int(np.prod(img_shape)), 512),
            # nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels=None):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        if labels.size(1) == 1:
            one_hot = torch.cuda.FloatTensor(labels.size(0), 10)
            labels = one_hot.scatter_(1, labels.data, 1)
            labels = labels.unsqueeze(1).expand(-1, int(img.size(0) / labels.size(0)), -1).reshape(img.size(0), -1)
            # labels.squeeze(1)
            # labels = self.label_embedding(labels)
            # labels = labels.expand(-1, int(img.size(0) / labels.size(0)), -1).reshape(img.size(0), -1)
        else:
            # labels = torch.cat(int(img.size(0) / labels.size(0)) * [labels])
            labels = labels.unsqueeze(1).expand(-1, int(img.size(0) / labels.size(0)), -1).reshape(img.size(0), -1)
        d_in = torch.cat((img.view(img.size(0), -1), labels), -1)
        # d_in = img.view(img.size(0), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# DeepSet Encoder
phi = SmallMNISTCNNPhi(dim=int(np.prod(img_shape)))
rho = SmallRho(input_size=10, output_size=10)
encoder1 = InvariantModel(phi=phi, rho=rho, set_size=opt.set_size, img_shape=img_shape)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    encoder1.cuda()
    # encoder2.cuda()
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(
#             ), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

train_db = MNISTSummation(set_size=opt.set_size, train=True, transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(),
                                                                                           transforms.Normalize([0.5], [0.5])]))
# train_db = Parametric(set_size=opt.set_size)
dataloader = torch.utils.data.DataLoader(train_db, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_E1 = torch.optim.Adam(encoder1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_E2 = torch.optim.Adam(encoder2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

tsne = TSNE(n_components=2, random_state=0)


# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Sample noise
#     z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
#     # Get labels ranging from 0 to n_classes for n rows
#     labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#     labels = Variable(LongTensor(labels))
#     gen_imgs = generator(z, labels)
#     save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    real_imgs = None
    encodings = None
    gen_imgs = None
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        set_size = imgs.shape[1]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size * set_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size * set_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs_prv = real_imgs
        real_imgs = Variable(imgs.type(FloatTensor))
        # labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        optimizer_E1.zero_grad()
        # optimizer_E2.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size * set_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        encodings_prv = encodings

        encodings = encoder1(real_imgs)
        # encodings.retain_grad()
        # encodings2 = encoder2(real_imgs)

        # Generate a batch of images
        # gen_imgs = generator(z, gen_labels)
        # gen_imgs = generator(z, labels)
        gen_imgs_prv = gen_imgs
        gen_imgs = generator(z, encodings)

        # OT Loss Regularization
        num_paths = 50
        src = np.random.randint(0, batch_size, size=num_paths)
        dest = np.random.randint(0, batch_size, size=num_paths)
        src_enc, dest_enc = encodings[src], encodings[dest]
        src_z, dest_z = z[src], z[dest]
        src_gen, dest_gen = gen_imgs[src].reshape(num_paths, -1), gen_imgs[dest].reshape(num_paths, -1)
        t = torch.rand(num_paths).unsqueeze(1).cuda()
        inter_enc = src_enc * t + dest_enc * (1 - t)
        inter_z = src_z * t + dest_z * (1 - t)
        inter_gen = generator(inter_z, inter_enc).reshape(num_paths, -1)
        gen_inter = src_gen * t + dest_gen * (1 - t)
        ot_loss = adversarial_loss(inter_gen, gen_inter)

        # for s, d in zip(src, dest):
        # tmp_imgs = gen_imgs.reshape(gen_imgs.shape[0], -1)
        # for i in range(tmp_imgs.shape[0]):
        #     for j in range(tmp_imgs.shape[1]):
        #         grd1 = torch.autograd.grad(tmp_imgs[i, j], encodings, create_graph=True)[0][int(i / batch_size)]
        #         for k in range(grd1.shape[0]):
        #             grd2 = torch.autograd.grad(grd1[k], encodings, retain_graph=True)[0][int(i / batch_size)]
        #             # ot_loss += sum([gr2.norm()**2 for gr2 in grd2])
        #             ot_loss += torch.norm(grd2)**2

        # Loss measures generator's ability to fool the discriminator
        # validity = discriminator(gen_imgs)
        # validity = discriminator(gen_imgs, gen_labels)
        # validity = discriminator(gen_imgs, labels)
        validity = discriminator(gen_imgs, encodings.detach())
        # validity = discriminator(gen_imgs, encodings)
        # validity = discriminator(gen_imgs, encodings2.detach())

        g_loss = adversarial_loss(validity, valid) + ot_loss
        # embed()
        # exit()

        # g_loss.backward()
        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        # optimizer_E1.step()

        # optimizer_E1.zero_grad()
        # encodings = encoder1(real_imgs)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        real_imgs = torch.reshape(real_imgs, (-1, *img_shape))
        # validity_real = discriminator(real_imgs)
        # validity_real = discriminator(real_imgs, labels)
        # validity_real = discriminator(real_imgs, encodings.detach())
        validity_real = discriminator(real_imgs, encodings)
        # validity_real = discriminator(real_imgs, encodings2)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        # validity_fake = discriminator(gen_imgs.detach())
        # validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        # validity_fake = discriminator(gen_imgs.detach(), labels)
        # validity_fake = discriminator(gen_imgs.detach(), encodings.detach())
        validity_fake = discriminator(gen_imgs.detach(), encodings)
        # validity_fake = discriminator(gen_imgs.detach(), encodings2)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        optimizer_E1.step()
        # optimizer_E2.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [OT loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), ot_loss.item())
        )

        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     # sample_image(n_row=10, batches_done=batches_done)
        #     save_image(real_imgs.data, "real_images_%d_%d/%d.png" % (batch_size, set_size, batches_done), nrow=8, normalize=True)
        #     save_image(gen_imgs.data, "gen_images_%d_%d/%d.png" % (batch_size, set_size, batches_done), nrow=8, normalize=True)

    save_image(real_imgs.data, "real_images_%d_%d/%d_1.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    save_image(gen_imgs.data, "gen_images_%d_%d/%d_1.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)

    # save_image(real_imgs_prv.data[:-set_size], "real_images_%d_%d/%d_0.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    # save_image(gen_imgs_prv.data[:-set_size], "gen_images_%d_%d/%d_0.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    # for i in np.arange(0.1, 1.0, 0.1):
    #     gen_imgs_inter = generator(z, ((1 - i) * encodings_prv[:-1] + i * encodings))
    #     save_image(gen_imgs_inter.data, "gen_images_%d_%d/%d_%f.png" % (opt.batch_size, set_size, epoch, i), nrow=8, normalize=True)

    if epoch % 5 == 0:
        X = np.empty([len(dataloader.dataset), opt.enc_dim])
        Y = np.empty(len(dataloader.dataset), dtype=int)

        for i, (imgs, labels) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(FloatTensor))
            encodings = encoder1(real_imgs)
            X[i * opt.batch_size:(i + 1) * opt.batch_size] = encodings.detach().cpu().numpy()
            Y[i * opt.batch_size:(i + 1) * opt.batch_size] = labels.squeeze().numpy()

        X_2d = tsne.fit_transform(X)

        target_ids = range(10)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_ids):
            plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
            # plt.scatter(X[Y == i, 0], X[Y == i, 1], c=c, label=label)

        plt.legend()
        plt.savefig("tsne/test_%d.png" % epoch)
