import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
# from sklearn import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing
import random
import ot
from tensorboardX import SummaryWriter
import datetime
from scipy.stats import pearsonr
from encoder import InvariantModel, SmallMNISTCNNPhi, SmallRho
from dataset import SciplexData
from sklearn.manifold import TSNE
import umap

import sciplex
from pathlib import Path
import seaborn as sns
import matplotlib.colors as colors

os.makedirs("images", exist_ok=True)
os.makedirs("dist", exist_ok=True)
os.makedirs("scatter", exist_ok=True)
os.makedirs("tsne", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_ot", type=float, default=0, help="regularization for ot loss")
parser.add_argument("--lambda_emb", type=float, default=0, help="regularization for ot loss")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=50, help="interval betwen image samples")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--num_paths", type=int, default=10, help="number of trajectories to sample for each start-end pair")
parser.add_argument("--enc_dim", type=int, default=10, help="output dim of the encoder")
parser.add_argument("--set_size", type=int, default=200, help="size of the sets")
parser.add_argument("--num_sets", type=int, default=30, help="number of sets per class")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)
img_shape = (100,)

cuda = True if torch.cuda.is_available() else False

# summary_writer = SummaryWriter(log_dir='/home/rishabh/distributional_encoder_cgan/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = SummaryWriter(log_dir='/home/rishabh/distributional_encoder_cgan/logs/ep%d_ot_%.2f_emb_%.2f' % (opt.n_epochs, opt.lambda_ot, opt.lambda_emb))


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
            # nn.Tanh()
        )

    def forward(self, z, labels, dest_labels=None):
        # labels = torch.repeat_interleave(labels, int(z.size(0) / labels.size(0)))
        labels = labels.unsqueeze(1).expand(-1, int(z.size(0) / labels.size(0)), -1).reshape(z.size(0), -1)
        if dest_labels != None:
            # p_dest = p_dest.unsqueeze(1).expand(-1, int(z.size(0) / p_dest.size(0))).reshape(z.size(0))
            # p_dest = torch.repeat_interleave(p_dest, int(z.size(0) / p_dest.size(0)))
            # dest_labels = torch.repeat_interleave(dest_labels, int(z.size(0) / dest_labels.size(0)))
            dest_labels = dest_labels.unsqueeze(1).expand(-1, int(z.size(0) / dest_labels.size(0)), -1).reshape(z.size(0), -1)
            # enc = 0.5 * self.label_emb(LongTensor([0] * labels.size(0))) + 0.5 * self.label_emb(LongTensor([1] * labels.size(0)))
            # enc = 0.5 * self.label_emb(labels) + 0.5 * self.label_emb(dest_labels)
            enc = 0.5 * labels + 0.5 * dest_labels
            gen_input = torch.cat((enc, z), -1)
        else:
            # gen_input = torch.cat((self.label_emb(labels), z), -1)
            gen_input = torch.cat((labels, z), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.enc_dim + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        # labels = torch.repeat_interleave(labels, int(img.size(0) / labels.size(0)))
        labels = labels.unsqueeze(1).expand(-1, int(img.size(0) / labels.size(0)), -1).reshape(img.size(0), -1)
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = torch.cat((img.view(img.size(0), -1), labels), -1)
        # img_flat = img.view(img.shape[0], -1)
        validity = self.model(d_in)
        return validity


# Loss weight for gradient penalty
# lambda_gp = 10
lambda_gp = 0.1
# lambda_ot = 1

# DeepSet Encoder
phi = SmallMNISTCNNPhi(dim=int(np.prod(img_shape)))
rho = SmallRho(input_size=10, output_size=opt.enc_dim)
encoder = InvariantModel(phi=phi, rho=rho, set_size=opt.set_size, img_shape=img_shape)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
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

train_db = SciplexData(set_size=opt.set_size, num_sets=opt.num_sets, train=True)
dataloader = torch.utils.data.DataLoader(train_db, batch_size=opt.batch_size, shuffle=True, drop_last=True)

# x, y = datasets.make_moons(n_samples=1000, noise=0.05)
# x = preprocessing.StandardScaler(with_std=False).fit_transform(x)
# x = preprocessing.MaxAbsScaler().fit_transform(x)

# def inf_train_gen():
#     scale = 2.
#     centers = [
#         (1, 0),
#         (-1, 0),
#         (0, 1),
#         (0, -1),
#         (1. / np.sqrt(2), 1. / np.sqrt(2)),
#         (1. / np.sqrt(2), -1. / np.sqrt(2)),
#         (-1. / np.sqrt(2), 1. / np.sqrt(2)),
#         (-1. / np.sqrt(2), -1. / np.sqrt(2))
#     ]
#     centers = [(scale * x, scale * y) for x, y in centers]
#     while True:
#         dataset = []
#         labels = []
#         for i in range(opt.batch_size):
#             # point = np.random.randn(2) * .02
#             point = np.random.randn(2) * .2
#             # center = random.choice(centers)
#             label = random.randint(0, 7)
#             point[0] += centers[label][0]
#             point[1] += centers[label][1]
#             dataset.append(point)
#             labels.append(label)
#         dataset = np.array(dataset, dtype='float32')
#         dataset /= 1.414  # stdev
#         labels = np.array(labels, dtype='int32')
#         yield dataset, labels


# data = inf_train_gen()

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

mse = nn.MSELoss()
tsne = TSNE(n_components=2, random_state=0)
umap = umap.UMAP()


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

batches_done = 0
d_rg = []
w_rg = []
diff = []
for epoch in range(opt.n_epochs):
    for b, (imgs, labels) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(LongTensor))

        # # perm = np.random.permutation(len(imgs))
        # batch = np.random.choice(len(x), opt.batch_size)
        # imgs, labels = x[batch], y[batch]
        # # imgs, labels = data.__next__()

        # batch_size = imgs.shape[0]

        # real_imgs = Variable((torch.from_numpy(imgs).type(Tensor)))
        # labels = Variable(torch.from_numpy(labels).type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_E.zero_grad()
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0] * imgs.shape[1], opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        gen_labels = labels

        encodings = encoder(real_imgs)

        # Generate a batch of images
        # fake_imgs = generator(z, gen_labels)
        fake_imgs = generator(z, encodings)
        real_imgs = torch.reshape(real_imgs, (-1, *img_shape))

        # Real images
        # real_validity = discriminator(real_imgs, labels)
        real_validity = discriminator(real_imgs, encodings)
        # Fake images
        # fake_validity = discriminator(fake_imgs, gen_labels)
        fake_validity = discriminator(fake_imgs.detach(), encodings)
        # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, encodings)
        # gradient_penalty = 0

        # OT penalty
        # ot_penalty = torch.mean(real_validity[y == 1]) - torch.mean(real_validity[y == 0])

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty + lambda_ot * ot_penalty

        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        # if i % opt.n_critic == 0:
        if b % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            # fake_imgs = generator(z, gen_labels)
            # embed()
            # exit()
            p_dest = torch.randperm(opt.batch_size)
            ids = (gen_labels != gen_labels[p_dest])
            # src_labels = gen_labels[ids]
            # dest_labels = gen_labels[p_dest][ids]
            src_enc = encodings[ids]
            dest_enc = encodings[p_dest][ids]

            # batch_size = src_labels.shape[0]
            batch_size = src_enc.shape[0]

            z = Variable(Tensor(np.random.normal(0, 1, (batch_size * opt.num_paths, opt.latent_dim))))

            fake_imgs_src = generator(z, src_enc)
            # enc_src = generator.label_emb(src_labels)
            # fake_imgs_dest = generator(z, 1 - gen_labels)
            fake_imgs_dest = generator(z, dest_enc)
            # enc_dest = generator.label_emb(dest_labels)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            # fake_validity = discriminator(fake_imgs, gen_labels)
            # fake_validity = (discriminator(fake_imgs_src, gen_labels) + discriminator(fake_imgs_dest, 1 - gen_labels)) / 2
            fake_validity = (discriminator(fake_imgs_src, src_enc.detach()) + discriminator(fake_imgs_dest, dest_enc.detach())) / 2

            # OT Loss
            # src = LongTensor([0] * opt.num_paths)
            # dest = LongTensor([1] * opt.num_paths)
            # src_enc, dest_enc = generator.label_emb(src), generator.label_emb(dest)
            # src_z = Variable(Tensor(np.random.normal(0, 1, (opt.num_paths, opt.latent_dim))))
            # src_gen, dest_gen = generator(src_z, src), generator(src_z, dest)
            # fake_imgs_mid = generator(z, gen_labels, flag=False)
            fake_imgs_mid = generator(z, src_enc, dest_enc)
            pos_dist = torch.zeros(batch_size)
            neg_dist = torch.zeros(batch_size)
            wass = [0] * batch_size
            for i in range(batch_size):
                src = fake_imgs_src[i * opt.num_paths: (i + 1) * opt.num_paths].reshape(opt.num_paths, -1)
                dest = fake_imgs_dest[i * opt.num_paths: (i + 1) * opt.num_paths].reshape(opt.num_paths, -1)
                # pos_dist[i] = torch.mean(torch.norm(src - dest, p=1, dim=1))
                pos_dist[i] = torch.mean(torch.norm(src - dest, p=2, dim=1))
                p = torch.randperm(opt.num_paths)
                # neg_dist[i] = torch.mean(torch.norm(src - dest[p], p=1, dim=1))
                neg_dist[i] = torch.mean(torch.norm(src - dest[p], p=2, dim=1))
                r, g = np.ones((opt.num_paths,)) / opt.num_paths, np.ones((opt.num_paths,)) / opt.num_paths
                M = ot.dist(src.cpu().detach().numpy(), dest.cpu().detach().numpy(), metric='euclidean')
                # M = ot.dist(src.cpu().detach().numpy(), dest.cpu().detach().numpy(), metric='minkowski', p=1)
                wass[i] = ot.emd2(r, g, M)
            euc = torch.norm(src_enc - dest_enc, p=2, dim=1)
            # ot_loss = pos_dist - neg_dist
            ot_loss = torch.sum(pos_dist) / torch.sum(neg_dist)
            # ot_loss = pos_dist
            emb_loss = mse(euc, pos_dist.cuda())

            g_loss = -torch.mean(fake_validity)
            g_loss_total = g_loss + opt.lambda_ot * ot_loss + opt.lambda_emb * emb_loss

            # g_loss.backward()
            g_loss_total.backward()
            optimizer_G.step()
            optimizer_E.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [OT loss: %f] [Emb loss: %f]"
                % (epoch, opt.n_epochs, b, len(dataloader), d_loss.item(), g_loss.item(), ot_loss.item(), emb_loss.item())
            )

            summary_writer.add_scalar('D_loss', d_loss.item(), epoch)
            summary_writer.add_scalar('G_loss', g_loss.item(), epoch)
            summary_writer.add_scalar('Contrastive_loss', ot_loss.item(), epoch)
            summary_writer.add_scalar('Pos_loss', torch.mean(pos_dist).item(), epoch)
            summary_writer.add_scalar('Emb_loss', emb_loss.item(), epoch)

            if batches_done % opt.sample_interval == 0:

                # # for c in range(opt.n_classes):
                # size0 = real_validity[y == 0].shape[0]
                # # size1 = fake_validity[y == 1].shape[0]
                # size1 = real_validity[y == 1].shape[0]
                # # val = torch.mean(real_validity[y == 0]) - torch.mean(fake_validity[y == 1])
                # val = torch.mean(real_validity[y == 0]) - torch.mean(real_validity[y == 1])
                # # r, g = np.ones((opt.batch_size,)) / opt.batch_size, np.ones((opt.batch_size,)) / opt.batch_size
                # # M = ot.dist(real_imgs[y == 0].cpu().detach().numpy(), fake_imgs[y == 1].cpu().detach().numpy(), metric='euclidean')
                # M = ot.dist(real_imgs[y == 0].cpu().detach().numpy(), real_imgs[y == 1].cpu().detach().numpy(), metric='euclidean')
                # r, g = np.ones((size0,)) / size0, np.ones((size1,)) / size1
                # wass = ot.emd2(r, g, M)
                # d_rg.append(val.cpu().detach().numpy())
                # w_rg.append(wass)
                # diff.append(wass - val.cpu().detach().numpy())
                # # break

                # plt.figure()
                # plt.plot(d_rg, label="Discriminator value function")
                # plt.plot(w_rg, label="True Wasserstein Distance")
                # plt.legend()
                # plt.xlabel("Training Time")
                # # plt.ylabel("Distances")
                # plt.savefig("dist/%d.png" % epoch)

                # plt.figure()
                # plt.plot(diff, label="Difference")
                # plt.legend()
                # plt.xlabel("Training Time")
                # # plt.ylabel("Distances")
                # plt.savefig("dist/diff_%d.png" % epoch)

                save_image(fake_imgs_src.data[:50:2], "images/src_%d.png" % epoch, nrow=5, normalize=True)
                save_image(fake_imgs_dest.data[:50:2], "images/dest_%d.png" % epoch, nrow=5, normalize=True)
                save_image(fake_imgs_mid.data[:50:2], "images/mid_%d.png" % epoch, nrow=5, normalize=True)
                # real_imgs = real_imgs.cpu().detach().numpy()
                # plt.figure()
                # plt.scatter(real_imgs[:, 0], real_imgs[:, 1], c=labels.cpu().detach().numpy())
                # plt.savefig("images/real_%d.png" % epoch)
                # plt.clf()

                # fake_imgs = fake_imgs_src.cpu().detach().numpy()
                # plt.figure()
                # plt.scatter(fake_imgs[:, 0], fake_imgs[:, 1], c=labels.cpu().detach().numpy())
                # plt.savefig("images/fake_%d.png" % epoch)
                # plt.clf()

                # plt.figure()
                # colors = cm.rainbow(np.linspace(0, 1, 3))
                # for fake_imgs, c in zip([fake_imgs_src, fake_imgs_mid, fake_imgs_dest], colors):
                #     fake_imgs = fake_imgs.cpu().detach().numpy()
                #     plt.scatter(fake_imgs[:, 0], fake_imgs[:, 1], color=c)
                # plt.savefig("images/traj_%d.png" % epoch)
                # plt.clf()

                # fake_imgs_src = fake_imgs_src.cpu().detach().numpy()
                # fake_imgs_dest = fake_imgs_dest.cpu().detach().numpy()
                # perm = np.random.randint(opt.batch_size * opt.num_paths, size=50)
                # plt.figure()
                # for i in perm:
                #     plt.plot([fake_imgs_src[i, 0], fake_imgs_dest[i, 0]], [fake_imgs_src[i, 1], fake_imgs_dest[i, 1]], 'k-')
                # # colors = [labels, 1 - labels]
                # colors = [labels.repeat_interleave(int(z.size(0) / labels.size(0))), labels[p_dest].repeat_interleave(int(z.size(0) / labels.size(0)))]
                # for fake_imgs, c in zip([fake_imgs_src, fake_imgs_dest], colors):
                #     plt.scatter(fake_imgs[:, 0], fake_imgs[:, 1], c=c.cpu().detach().numpy())
                # plt.savefig("images/traj_%d.png" % epoch)
                # plt.clf()
                corr, _ = pearsonr(wass, pos_dist.cpu().detach().numpy())
                summary_writer.add_scalar('Correlation', corr, epoch)

                # plt.figure()
                # plt.scatter(wass, pos_dist.cpu().detach().numpy())
                # plt.xlabel("True Wass. Distance")
                # plt.ylabel("Appx Wass. Distance")
                # plt.savefig("images/dist_%d.png" % epoch)
                # plt.clf()

                plt.figure()
                plt.scatter(wass, euc.cpu().detach().numpy())
                plt.xlabel("Wass. Distance")
                plt.ylabel("Euc. Distance")
                plt.savefig("scatter/scatter_%d.png" % epoch)
                plt.clf()

            batches_done += opt.n_critic

    if epoch % 10 == 0:
        X = np.empty([len(dataloader.dataset), opt.enc_dim])
        Y = np.empty(len(dataloader.dataset), dtype=int)
        treatments = dataloader.dataset.treatments
        doses = dataloader.dataset.doses

        for i, (imgs, labels) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))
            encodings = encoder(real_imgs)
            X[i * opt.batch_size:(i + 1) * opt.batch_size] = encodings.detach().cpu().numpy()
            Y[i * opt.batch_size:(i + 1) * opt.batch_size] = labels.squeeze().numpy()

        # X_2d = tsne.fit_transform(X)
        encodings = encoder(dataloader.dataset.data.type(Tensor))
        X_2d = tsne.fit_transform(encodings.detach().cpu().numpy())
        # X_2d = umap.fit_transform(encodings.detach().cpu().numpy())
        T = [treatments[i] for i in dataloader.dataset.targets]
        D = [doses[i] for i in dataloader.dataset.targets]

        target_ids = range(opt.n_classes)
        # plt.figure(figsize=(6, 5))
        plt.figure()
        # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
        # for i, c, label in zip(target_ids, colors, conditions):
        #     plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
        #     # plt.scatter(X[Y == i, 0], X[Y == i, 1], c=c, label=label)
        # norm = colors.LogNorm()
        # sns.scatterplot(X_2d[:, 0], X_2d[:, 1], hue=dataloader.dataset.labels, palette=)
        # embed()
        # exit()

        # sns.scatterplot(X_2d[:, 0], X_2d[:, 1], hue=dataloader.dataset.labels, palette="deep")
        sns.scatterplot(X_2d[:, 0], X_2d[:, 1], hue=T, style=D)

        plt.legend(bbox_to_anchor=(1.05, 1), ncol=2)
        plt.savefig("tsne/tsne_%d.png" % epoch, bbox_inches="tight")
