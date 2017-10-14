import torch
import torch.nn as nn
from torch.autograd import Variable as V

class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, drop_rate = 0.5, tie_weights = False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(drop_rate)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type is 'LSTM':
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout = drop_rate)
        elif rnn_type is 'GRU':
            self.rnn = nn.GRU(ninp, nhid, nlayers, dropout = drop_rate)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity = nonlinearity, dropout = drop_rate)

        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if ninp != nhid:
                raise AssertionError
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (V(weight.new(self.nlayers, batch_size, self.nhid).zero_()),
                    V(weight.new(self.nlayers, batch_size, self.nhid).zero_()))
        else:
            return V(weight.new(self.nlayers, batch_size, self.nhid).zero_())
