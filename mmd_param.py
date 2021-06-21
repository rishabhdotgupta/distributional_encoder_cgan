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
# from encoder import InvariantModel, SmallMNISTCNNPhi, SmallRho
from param_encoder import InvariantModel, SmallMNISTCNNPhi, SmallRho
from IPython import embed
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
from scipy.stats import pearsonr
import ot
import time
from MMD import mix_rbf_mmd2

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--enc_dim", type=int, default=2, help="output dim of the encoder")
parser.add_argument("--reg1", type=float, default=0, help="regularization for ot loss")
parser.add_argument("--reg2", type=float, default=0, help="regularization for new loss")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--set_size", type=int, default=1, help="size of the sets")
parser.add_argument("--num_paths", type=int, default=100, help="number of trajectories to sample")
opt = parser.parse_args()
print(opt)

# os.makedirs("gen_images_%d_%d/" % (opt.batch_size, opt.set_size), exist_ok=True)
# os.makedirs("real_images_%d_%d/" % (opt.batch_size, opt.set_size), exist_ok=True)
os.makedirs("tsne_param/", exist_ok=True)
os.makedirs("corr_param/", exist_ok=True)
os.makedirs("losses_param/", exist_ok=True)
os.makedirs("gen_param/", exist_ok=True)

# img_shape = (opt.channels, opt.img_size, opt.img_size)
img_shape = (opt.channels,)

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
            # *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            # nn.Tanh()
        )

    def forward(self, noise, labels):
        # if labels.size(1) == 1:
        #     one_hot = torch.cuda.FloatTensor(labels.size(0), opt.enc_dim)
        #     labels = one_hot.scatter_(1, labels.data, 1)
        #     labels = labels.unsqueeze(1).expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
        #     # labels.squeeze(1)
        #     # labels = self.label_emb(labels)
        #     # labels = labels.expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
        # else:
        #     # Concatenate label embedding and image to produce input
        #     # gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #     # labels = torch.cat(int(noise.size(0) / labels.size(0)) * [labels])
        #     labels = labels.unsqueeze(1).expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
        labels = labels.unsqueeze(1).expand(-1, int(noise.size(0) / labels.size(0)), -1).reshape(noise.size(0), -1)
        gen_input = torch.cat((labels, noise), -1)
        # gen_input = noise
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# DeepSet Encoder
phi = SmallMNISTCNNPhi(dim=int(np.prod(img_shape)))
rho = SmallRho(input_size=10, output_size=opt.enc_dim)
encoder1 = InvariantModel(phi=phi, rho=rho, set_size=opt.set_size, img_shape=img_shape)

# Initialize generator and discriminator
generator = Generator()

if cuda:
    encoder1.cuda()
    generator.cuda()
    adversarial_loss.cuda()

# # Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
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

# train_db = MNISTSummation(set_size=opt.set_size, train=True, transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(),
                                                                                           # transforms.Normalize([0.5], [0.5])]))
train_db = Parametric(set_size=opt.set_size)
dataloader = torch.utils.data.DataLoader(train_db, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_E1 = torch.optim.Adam(encoder1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_E2 = torch.optim.Adam(encoder2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


sinkhorn1 = SamplesLoss(loss="sinkhorn", scaling=0.9, p=1, debias=False)
# sinkhorn2 = SamplesLoss(loss="sinkhorn", scaling=0.9, p=2, debias=False)
tsne = TSNE(n_components=2, random_state=0)
sigma_list = np.logspace(-3, 2, 10)

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
gen = []
ot1 = []
ot2 = []
dif1 = []
dif2 = []
total = 0
total_time = []
for epoch in range(opt.n_epochs):
    sink1 = []
    euc = []

    encoder1.train()
    # generator.train()
    real_imgs = None
    encodings = None
    gen_imgs = None
    labels_old = None
    gen_loss_total = 0
    ot_loss_total = 0
    loss_2_total = 0
    for i, (imgs, labels) in enumerate(dataloader):
        start = time.time()

        batch_size = imgs.shape[0]
        set_size = imgs.shape[1]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size * set_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size * set_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs_prv = real_imgs
        real_imgs = Variable(imgs.type(FloatTensor))
        labels_prv = labels_old
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        optimizer_E1.zero_grad()
        # optimizer_E2.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size * set_size, opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        # encodings_prv = encodings
        if real_imgs_prv is not None:
            encodings_prv = encoder1(real_imgs_prv)
        else:
            encodings_prv = encodings
        encodings = encoder1(real_imgs)
        # encodings.retain_grad()

        # Generate a batch of images
        # gen_imgs = generator(z, gen_labels)
        if encodings_prv is not None:
            gen_imgs_prv = generator(z, encodings_prv[:batch_size])
        else:
            gen_imgs_prv = gen_imgs
        # gen_imgs = generator(z, labels)
        gen_imgs = generator(z, encodings)

        # OT Loss
        eps = 0.01
        src = np.random.randint(0, batch_size, size=opt.num_paths)
        dest = np.random.randint(0, batch_size, size=opt.num_paths)
        src_enc, dest_enc = encodings[src], encodings[dest]
        src_z = z[src * opt.set_size]
        # src_gen, dest_gen = gen_imgs[src].reshape(num_paths, -1), gen_imgs[dest].reshape(num_paths, -1)
        src_gen, dest_gen = gen_imgs[src * opt.set_size], generator(src_z, dest_enc)
        t = torch.rand(opt.num_paths).unsqueeze(1).cuda()
        t_eps = t + eps
        # gen_inter = src_gen * t + dest_gen * (1 - t)
        inter_enc = src_enc * t + dest_enc * (1 - t)
        inter_enc_eps = src_enc * t_eps + dest_enc * (1 - t_eps)
        inter_gen = generator(src_z, inter_enc).reshape(opt.num_paths, -1)
        # inter_enc_eps = inter_enc + eps
        inter_gen_eps = generator(src_z, inter_enc_eps).reshape(opt.num_paths, -1)
        # ot_loss = adversarial_loss(inter_gen, gen_inter)
        ot_loss = adversarial_loss(inter_gen_eps, inter_gen) / eps

        src_z = Variable(FloatTensor(np.random.normal(0, 1, (opt.num_paths * opt.num_paths, opt.latent_dim))))
        src_gen, dest_gen = generator(src_z, src_enc), generator(src_z, dest_enc)
        src_gen = src_gen.reshape(opt.num_paths, opt.num_paths)
        dest_gen = dest_gen.reshape(opt.num_paths, opt.num_paths)
        gen_dist = torch.norm(src_gen - dest_gen, p=1, dim=1) / opt.num_paths
        enc_dist = torch.norm(src_enc - dest_enc, p=2, dim=1)
        loss_2 = adversarial_loss(enc_dist, gen_dist)

        gen_imgs = gen_imgs.reshape(batch_size, set_size, -1)
        g_loss_1 = 0
        for j in range(batch_size):
            g_loss_1 += mix_rbf_mmd2(gen_imgs[j].squeeze(0), real_imgs[j].squeeze(0), sigma_list = sigma_list)
        g_loss = g_loss_1 + opt.reg1 * ot_loss + opt.reg2 * loss_2
        gen_loss_total += g_loss_1.detach().cpu().numpy()
        ot_loss_total += ot_loss.detach().cpu().numpy()
        loss_2_total += loss_2.detach().cpu().numpy()

        g_loss.backward()
        optimizer_G.step()
        optimizer_E1.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [OT loss: %f] [Loss 2: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item(), ot_loss.item(), loss_2.item())
        )

        end = time.time()
        total += end - start

        sink_dist = sinkhorn1(imgs[src],imgs[dest]).to("cuda")
        sink_dist[src == dest] = 0
        euc_dist = torch.norm(encodings[src]-encodings[dest], p=2, dim=1)
        sink1.extend(sink_dist[src != dest].cpu().detach().numpy())
        euc.extend(euc_dist[src != dest].cpu().detach().numpy())

        labels_old = labels

    gen.append(gen_loss_total)
    ot1.append(ot_loss_total)
    ot2.append(loss_2_total)

    total_time.append(total)
    print(total)

    corr, _ = pearsonr(sink1,euc)
    print(corr)
    dif1.append(corr)

    plt.figure()
    plt.plot(total_time, dif1)
    plt.savefig("corr_param/corr_%d.png" % epoch)

    # save_image(real_imgs.data, "real_images_%d_%d/%d_1.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    # save_image(gen_imgs.data, "gen_images_%d_%d/%d_1.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    real_imgs_1 = real_imgs.reshape(-1, opt.set_size, 1)
    gen_imgs_1 = gen_imgs.reshape(-1, opt.set_size, 1)
    real_imgs_2 = real_imgs_prv.reshape(-1, opt.set_size, 1)
    gen_imgs_2 = gen_imgs_prv.reshape(-1, opt.set_size, 1)
    plt.figure()
    plt.subplot(211)
    plt.hist(real_imgs_1[0].cpu().numpy(), alpha=0.5, bins=50, range=[-5.0, 5.0], label='real0')
    plt.hist(real_imgs_2[0].detach().cpu().numpy(), alpha=0.5, bins=50, range=[-5.0, 5.0], label='real1')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.hist(gen_imgs_1[0].detach().cpu().numpy(), alpha=0.5, bins=50, range=[-5.0, 5.0], label='generated0')
    plt.hist(gen_imgs_2[0].detach().cpu().numpy(), alpha=0.5, bins=50, range=[-5.0, 5.0], label='generated1')
    # gen_imgs_inter = generator(z, (0.5 * encodings_prv[:8] + 0.5 * encodings))
    # gen_imgs_inter = gen_imgs_inter.reshape(-1,opt.set_size,1)
    # plt.hist(gen_imgs_inter[0].detach().cpu().numpy(), alpha=0.5, bins=50, range=[-5.0, 5.0], label='generated_mid')
    for t in np.arange(0.25, 1.0, 0.25):
        gen_imgs_inter = generator(z, (t * encodings_prv[:batch_size] + (1-t) * encodings))
        gen_imgs_inter = gen_imgs_inter.reshape(-1,opt.set_size,1)
        plt.hist(gen_imgs_inter[0].detach().cpu().numpy(), alpha=0.5, bins=50, range=[-5.0, 5.0], label='generated_%.2f' % t)
    plt.legend(loc='upper right')
    plt.savefig("gen_param/test_%d.png" % epoch)
    print(labels[0], labels_prv[0])

    
    generator.eval()

    fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [1, 2, 1]}, sharex=True)
    plt.setp(axs, xlim=(-4,4))

    axs[0].hist(gen_imgs_2[0].detach().cpu().numpy(), alpha=0.5, bins=20)
    axs[0].yaxis.set_visible(False)

    ids = np.random.randint(0, len(z), size=20)
    for i in ids:
        x = []
        T = []
        for t in np.arange(0.01, 1.0, 0.01):
            gen_imgs_inter = generator(z[i].unsqueeze(0), (t * encodings_prv[0].unsqueeze(0) + (1-t) * encodings[0].unsqueeze(0)))
            x.append(gen_imgs_inter.squeeze(0).detach().cpu().numpy())
            T.append(t)
        axs[1].plot(x,T,'-',color='darkgrey')
    axs[1].set_ylabel("t")

    axs[2].hist(gen_imgs_1[0].detach().cpu().numpy(), alpha=0.5, bins=20)
    axs[2].set_xlabel("x")
    axs[2].yaxis.set_visible(False)
    fig.savefig("gen_param/traj_%d.png" % epoch)

    # save_image(real_imgs_prv.data[:-set_size], "real_images_%d_%d/%d_0.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    # save_image(gen_imgs_prv.data[:-set_size], "gen_images_%d_%d/%d_0.png" % (opt.batch_size, set_size, epoch), nrow=8, normalize=True)
    # for i in np.arange(0.1, 1.0, 0.1):
    #     gen_imgs_inter = generator(z, ((1 - i) * encodings_prv[:-1] + i * encodings))
    #     save_image(gen_imgs_inter.data, "gen_images_%d_%d/%d_%f.png" % (opt.batch_size, set_size, epoch, i), nrow=8, normalize=True)

    if epoch % 5 == 4:
        X = np.empty([len(dataloader.dataset), opt.enc_dim])
        Y = np.empty(len(dataloader.dataset), dtype=int)
        start = 0

        encoder1.eval()
        for i, (imgs, labels) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(FloatTensor))
            batch_size = real_imgs.shape[0]
            encodings_full = encoder1(real_imgs)
            X[start:start + batch_size] = encodings_full.detach().cpu().numpy()
            Y[start:start + batch_size] = labels.squeeze().numpy()
            start += batch_size

        # X_2d = tsne.fit_transform(X)

        target_ids = range(10)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_ids):
            # plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
            plt.scatter(X[Y == i, 0], X[Y == i, 1], c=c, label=label)

        plt.legend()
        plt.savefig("tsne_param/test_%d.png" % epoch)

plt.figure()
plt.plot(np.log(gen), label="Generator Loss")
plt.legend()
plt.savefig("losses_param/gen_%d.png" % epoch)
plt.figure()
plt.plot(ot1, label="OT Loss")
plt.legend()
plt.savefig("losses_param/ot1_%d.png" % epoch)
plt.figure()
plt.plot(ot2, label="New Loss")
plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
plt.savefig("losses_param/ot2_%d.png" % epoch)

torch.save(encoder1.state_dict(), 'cgan_encoder_param_500_10.pkl')
encoder1.eval()

# plt.figure()
# # plt.plot(dif)
# rn_avg = np.convolve(dif1, np.ones(5)/5, mode='valid')
# plt.plot(rn_avg)
# rn_avg2 = np.convolve(dif2, np.ones(5)/5, mode='valid')
# plt.plot(rn_avg2)
# plt.savefig("sinkhorn/corr_final.png")

sink1 = []
# sink2 = []
wass1 = []
euc = []
cor = 0
for i, (imgs, labels) in enumerate(dataloader):
    real_imgs = Variable(imgs.type(FloatTensor))
    batch_size = real_imgs.shape[0]
    encodings = encoder1(real_imgs)
    src = np.random.randint(0, batch_size, size=opt.num_paths)
    dest = np.random.randint(0, batch_size, size=opt.num_paths)
    sink_dist1 = sinkhorn1(real_imgs[src], real_imgs[dest]).to("cuda")
    sink_dist1[src == dest] = 0
    # sink_dist2 = sinkhorn2(real_imgs[src],real_imgs[dest]).to("cuda")
    # sink_dist2 [src == dest] = 0
    for i in range(opt.num_paths):
        if src[i] != dest[i]:
            wass1.append(ot.emd2_1d(real_imgs[src[i]].cpu().numpy(), real_imgs[dest[i]].cpu().numpy(), metric='euclidean'))
    euc_dist = torch.norm(encodings[src] - encodings[dest], p=2, dim=1)
    sink1.extend(sink_dist1[src != dest].cpu().detach().numpy())
    # sink2.extend(sink_dist2[src != dest].cpu().detach().numpy())
    euc.extend(euc_dist[src != dest].cpu().detach().numpy())
    # corr, _ = pearsonr(sink_dist.cpu().detach().numpy(),euc_dist.cpu().detach().numpy())
    # cor += corr

corr1, _ = pearsonr(sink1, euc)
# corr2, _ = pearsonr(sink2,euc)
corr2, _ = pearsonr(wass1, euc)
print(corr1, corr2)
plt.figure()
plt.scatter(sink1, euc)
plt.xlabel("Sinkhorn Distance")
plt.ylabel("Euclidean Distance")
plt.savefig("corr_param/scatter_1.png")
plt.figure()
plt.scatter(wass1, euc)
plt.xlabel("Wasserstein Distance")
plt.ylabel("Euclidean Distance")
plt.savefig("corr_param/scatter_2.png")
