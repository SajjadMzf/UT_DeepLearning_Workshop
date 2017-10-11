import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_input=28*28, num_class=10, num_hidden_layer = 1, num_hidden_unit = [100], activation = 'relu', dropout = True, drop_rate = 0.5):
        super(Model, self).__init__()
        self.num_input = num_input
        self.num_class =  num_class
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_unit = num_hidden_unit
        self.activation = activation

        layers = []
        layers.append(nn.Linear(num_input, num_hidden_unit[0]))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(negative_slope = 0.01, inplace=False))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sig':
            layers.append(nn.Sigmoid())

        for i in range(1,num_hidden_layer):
            layers.append(nn.Linear(num_hidden_unit[i-1],num_hidden_unit[i]))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky':
                layers.append(nn.LeakyReLU(negative_slope = 0.01, inplace=False))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sig':
                layers.append(nn.Sigmoid())
        if dropout:
            layers.append(nn.Dropout(p=drop_rate))
        layers.append(nn.Linear(num_hidden_unit[-1], num_class))

        self.linears = nn.ModuleList(layers)

    def forward(self, input):
        input = input.view(-1, self.num_input)
        out = input
        for l in self.linears:
            out = l(out)
        return out
