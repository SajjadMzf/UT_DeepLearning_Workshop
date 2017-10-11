import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torchvision import datasets as dsets
from torchvision import transforms as trans
from torch.autograd import Variable as V
import numpy as np
import pdb

# ------------- (Start) Hyper Parameters ---------------
bs = 32 # Batch Size
learning_rate = 1e-3
wd = 1e-4 # weight_decay
itr = 10
cuda = False

# ------------- (Start) Tensorboard in Pytorch :P ------
from logger import Logger
logger = Logger('./logs')
# ------------- (End) Tensorboard in Pytorch :P --------

# ------------- (End) Hyper Parameters ---------------

torch.manual_seed(0)
if torch.cuda.is_available() and cuda:
    torch.cuda.manual_seed_all(0)
    FloatType = torch.cuda.FloatTensor
    LongType = torch.cuda.LongTensor
else:
    FloatType = torch.FloatTensor
    LongType = torch.LongTensor
# ------------- (End) Hyper Parameters ---------------

# Define transformation
transforms = trans.Compose([trans.ToTensor(), trans.Normalize(mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5)),])

# Loading dataset
train_data = dsets.CIFAR10(root='./data', train = True, transform = transforms, download = True)
test_data = dsets.CIFAR10(root='./data', train = False, transform = transforms, download = True)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size = bs, shuffle = True, num_workers = 1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, shuffle = False, num_workers = 1)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight.data)
        m.bias.data.normal_(mean=0,std=1e-2)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.uniform_()
        m.bias.data.zero_()

# Model and Optimizer definition
model = Model(num_class=10, drop_rate = 0.5)

if cuda:
    model = model.cuda()

model.apply(weights_init)
optimizer = optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay = wd)
criterion = torch.nn.CrossEntropyLoss()

def train_model(model, optimizer, train_loader, criterion, epoch, vis_step = 20):
    model.train(mode = True)
    num_hit = 0
    total = len(train_loader.dataset)
    num_batch = np.ceil(total/bs)
    # Training Phase on train dataset
    for batch_idx, (image, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        image, labels = V(image.type(FloatType)), V(labels.type(LongType))
        output = model(image)
        loss = criterion(output, labels)

        if batch_idx % vis_step == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0]))
        loss.backward()
        optimizer.step()
    # Validation Phase on train dataset
    for batch_idx, (image, labels) in enumerate(train_loader):
        image, labels = V(image.type(FloatType)), V(labels.type(LongType))
        output = model(image)
        _ , pred_label = output.data.max(dim=1)
        num_hit += (pred_label == labels.data).sum()
    train_accuracy = (num_hit / total)
    print("Epoch: {}, Training Accuracy: {:.2f}%".format(epoch, 100. * train_accuracy))

    # ----------Tensorboard Computation -----------------
    info = {
        'accuracy-train': train_accuracy * 100.
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    # ----------Tensorboard Computation -----------------

    return 100. * train_accuracy

def eval_model(model, test_loader, epoch):
    model.train(mode = False)
    num_hit = 0
    total = len(test_loader.dataset)
    for batch_idx, (image, labels) in enumerate(test_loader):
        image, labels = V(image), V(labels)
        output = model(image)
        _ , pred_label = output.data.max(dim=1)
        num_hit += (pred_label == labels.data).sum()
    test_accuracy = (num_hit / total)
    print("Epoch: {}, Testing Accuracy: {:.2f}%".format(epoch, 100. * test_accuracy))
    # ----------Tensorboard Computation -----------------
    info = {
        'accuracy-test': test_accuracy * 100.
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    # ----------Tensorboard Computation -----------------
    return 100. * test_accuracy

train_acc = []
test_acc = []

for epoch in range(itr):
    tr_acc = train_model(model, optimizer, train_loader, criterion, epoch)
    ts_acc = eval_model(model, test_loader, epoch)
    train_acc.append(tr_acc)
    test_acc.append(ts_acc)

# import matplotlib.pyplot as plt
# plt.figure(num=1)
# plt.subplot(1,2,1)
# plt.plot(train_acc, 'r-')
# plt.xlabel("Epochs")
# plt.ylabel("Training Accuracy")
# plt.xlim(0, len(train_acc))
# plt.ylim(0, 100)
# plt.subplot(1,2,2)
# plt.plot(test_acc, 'b-')
# plt.xlabel("Epochs")
# plt.ylabel("Testing Accuracy")
# plt.xlim(0, len(train_acc))
# plt.ylim(0, 100)
# plt.show()
