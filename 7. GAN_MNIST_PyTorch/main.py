import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as trans
import torch.nn as nn

mb_size = 64
Z_dim = 100
h_dim = 128
c = 0
lr = 1e-3
X_dim = 28*28
y_dim = 10
niter = 10000

transforms = trans.Compose([trans.ToTensor(),])
train_data = dset.MNIST(root='./data', train = True, transform = transforms, download = False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = mb_size, shuffle = True, num_workers = 1, drop_last = True)



def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight.data)
        m.bias.data.normal_(mean=0,std=1e-2)

""" ==================== GENERATOR ======================== """

class Generator(nn.Module):
    """docstring for Generator."""
    def __init__(self, Z_dim, h_dim, X_dim):
        super(Generator, self).__init__()
        self.Z_dim = Z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.map1 = nn.Linear(Z_dim, h_dim)
        self.map2 = nn.Linear(h_dim, X_dim)
    def forward(self, inp):
        X = F.relu(self.map1(inp))
        X = F.sigmoid(self.map2(X))
        return X

""" ==================== DISCRIMINATOR ======================== """

class Discriminator(nn.Module):
    """docstring for Generator."""
    def __init__(self, X_dimm ,h_dim):
        super(Discriminator, self).__init__()
        self.X_dim = X_dim
        self.h_dim = h_dim

        self.map1 = nn.Linear(X_dim, h_dim)
        self.map2 = nn.Linear(h_dim, 1)
    def forward(self, inp):
        X = F.relu(self.map1(inp))
        X = F.sigmoid(self.map2(X))
        return X

""" ===================== TRAINING ======================== """

netD = Discriminator(X_dim, h_dim)
print(netD)
netG = Generator(Z_dim, h_dim, X_dim)
print(netG)

G_solver = optim.Adam(netG.parameters(), lr=1e-3)
D_solver = optim.Adam(netD.parameters(), lr=1e-3)

critetion = nn.BCELoss()

def train(netD, D_solver, netG, G_solver, critetion,epoch, train_loader, c):
    ones_label = Variable(torch.ones(mb_size,1))    # True label
    zeros_label = Variable(torch.zeros(mb_size,1))  # False label

    for index, (X, _) in enumerate(train_loader):
        z = Variable(torch.randn(mb_size, Z_dim))
        X = Variable(X.view(X.size(0), -1)) # (batch_size, 28*28)

        # Training Disc.
        D_solver.zero_grad()
        G_sample = netG(z)
        D_real = netD(X)
        D_fake = netD(G_sample)

        D_loss_real = critetion(D_real, ones_label)
        D_loss_fake = critetion(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        D_solver.step()


        # Training Genertor
        G_solver.zero_grad()
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = netG(z)
        D_fake = netD(G_sample)
        G_loss = critetion(D_fake, ones_label)
        G_loss.backward()
        G_solver.step()

        if index % 200 == 0:
            print('Epoch-{}; Index-{}; D_loss: {}; G_loss: {}'.format(epoch, index, D_loss.data[0], G_loss.data[0]))

            samples = netG(z).data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}.png'.format(str(epoch + index).zfill(3)), bbox_inches='tight')
            c += 1
            plt.close(fig)

for epoch in range(niter):
    train(netD, D_solver, netG, G_solver, critetion, epoch, train_loader, c)
