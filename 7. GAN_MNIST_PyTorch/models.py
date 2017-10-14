import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size = 1, num_hidden_layer = 2, num_hidden_unit = [4, 4], num_output = 1):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_unit = num_hidden_unit

        layer = []
        if num_hidden_layer == 0:
            layer.append(nn.Linear(input_size, num_output))
        else:
            layer.append(nn.Linear(input_size, num_hidden_unit[0]))
            layer.append(nn.ELU())

            for i in range(1, len(num_hidden_unit)):
                layer.append(nn.Linear(num_hidden_unit[i-1], num_hidden_unit[i]))
                layer.append(nn.ELU())

            layer.append(nn.Linear(num_hidden_unit[-1], num_output))

        self.layer = nn.ModuleList(layer)

    def forward(self, input):

        out = input
        for l in self.layer:
            out = l(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_size = 1, num_hidden_layer = 3, num_hidden_unit = [8, 8, 8], num_output = 1):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_unit = num_hidden_unit

        layer = []
        if num_hidden_layer == 0:
            layer.append(nn.Linear(input_size, num_output))
        else:
            layer.append(nn.Linear(input_size, num_hidden_unit[0]))
            layer.append(nn.ELU())

            for i in range(1, len(num_hidden_unit)):
                layer.append(nn.Linear(num_hidden_unit[i-1], num_hidden_unit[i]))
                layer.append(nn.ELU())

            layer.append(nn.Linear(num_hidden_unit[-1], num_output))

        layer.append(nn.Sigmoid())
        self.layer = nn.ModuleList(layer)

    def forward(self, input):

        out = input
        for l in self.layer:
            out = l(out)
        return out
