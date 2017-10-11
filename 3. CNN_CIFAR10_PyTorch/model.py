import torch.nn as nn
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self,
            num_class=10,
            drop_rate = 0.5):
        super(Model, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5,stride=1,padding=0,bias=False)),
            ('bn1', nn.BatchNorm2d(num_features = 6)),
            ('rl1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1,padding=0,bias=False)),
            ('bn2', nn.BatchNorm2d(num_features = 16)),
            ('rl2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2))
        ]))
        self.classifer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16*5*5,120)),
            ('drop1', nn.Dropout(drop_rate)),
            ('rl3', nn.ReLU()),
            ('fc2', nn.Linear(120,80)),
            ('drop2', nn.Dropout(drop_rate)),
            ('rl4', nn.ReLU()),
            ('fc3', nn.Linear(80,num_class))
        ]))

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out
