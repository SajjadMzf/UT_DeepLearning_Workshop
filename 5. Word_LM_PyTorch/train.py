import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import data as dataloader
from model import RNNModel
import torch.optim as optim
import pdb


""" ----------- Parameters Initialization------------"""
data = 'tinyshakespeare.txt'
rt = "LSTM"
embedding_size = 200
number_hidden = 200
number_layer = 2
learning_rate = 0.001  # Initial Learning Rate
clip = 0.25  # Gradient Clipping
epochs = 40
batch_size = 20
sequence_length = 20
dropout = 0.2   # Drop out rate
tied = True
seed = 0
cuda = False
step_vis = 50
save_model = True
words = 1000
temperature = 50    # Driversity

cuda = True

torch.manual_seed(seed)
if torch.cuda.is_available() and cuda:
    torch.cuda.manual_seed_all(seed)
    FloatType = torch.cuda.FloatTensor
    LongType = torch.cuda.LongTensor
else:
    FloatType = torch.FloatTensor
    LongType = torch.LongTensor


""" ----------- Initial Setting ------------"""
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda:
        print("You have cuda device, so you should probably run with")
    else:
        torch.cuda.manual_seed_all(seed)


""" ----------- Loading Data ------------"""
corpus = dataloader.Corpus(data)    # Convert text to ids

def create_batch(data, batch_size):
    number_batch = data.size(0) // batch_size
    data = data.narrow(dimension = 0, 
        start = 0, 
        length = number_batch * batch_size) 
    data = data.view(batch_size, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

train_data = create_batch(corpus.train, batch_size)

""" ----------- Model Creation ------------"""
number_tokens = len(corpus.dictionary)  # Number of unique word in our corpus

model = RNNModel(rnn_type = rt,
                ntoken = number_tokens,
                ninp = embedding_size,
                nhid = number_hidden,
                nlayers = number_layer,
                drop_rate = dropout,
                tie_weights = tied)

if cuda and torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr = learning_rate)

""" ----------- Training Code ------------"""
def detach_hidden(h): # detach from distant history
    if type(h) == V:
        return V(h.data)
    else:
        return tuple(detach_hidden(v) for v in h)


def get_batch(source, i, sequence_length):
    seq_len = min(sequence_length, len(source) - 1 - i)
    # torch.cat([data.data.view(-1).unsqueeze(-1), target.data.unsqueeze(-1)], dim=1)
    data = V(source[i:i+seq_len]).type(LongType)
    target = V(source[i+1:i+1+seq_len].view(-1)).type(LongType)
    return data, target

### LSTM Input and Output
"""
input (seq_len, batch, input_size): tensor containing the features of the input sequence. 
h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
c_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.

output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t. 
h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
"""

### GRU Input and Output
"""
input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence.
h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.

output (seq_len, batch, hidden_size * num_directions): tensor containing the output features h_t from the last layer of the RNN, for each t.
h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
"""

### RNN Input and Output
"""
input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence.
h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.

output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_k) from the last layer of the RNN, for each k. 
h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for k=seq_len.




"""




def train(model,
        optimizer,
        criterion,
        corpus,
        batch_size,
        train_data,
        sequence_length,
        clip,
        step_vis,
        epoch,
        learning_rate):
    model.train(True)
    total_loss = 0
    start_time = time.time()
    number_tokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, sequence_length)):
        data, targets = get_batch(train_data, i, sequence_length)
        hidden = detach_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, number_tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.data
        if batch % step_vis == 0 and batch > 0:
            current_loss = total_loss[0]  / step_vis
            elapsed_time = time.time() - start_time
            print('\n| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f}'.format(
                    epoch, batch, len(train_data) // sequence_length, learning_rate,
                    elapsed_time * 1000 / step_vis, current_loss))
            total_loss = 0
            start_time = time.time()

def generate(model, words, temperature, corpus):
    model.train(False)
    number_tokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = V(torch.rand(1,1).mul(number_tokens).type(LongType), volatile=True)
    for i in range(words):
        output , hidden = model(input, hidden)
        words_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(words_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        word = word.replace('<eos>','\n')
        # if ':' in word:
        #     word = '\n\n' + word
        print("%s "%(word), end='')
        # outf.write(word + ('\n' if i % 20 == 19 else ' '))
try:
    for epoch in range(1, epochs+1):
        train(model,
                optimizer,
                criterion,
                corpus,
                batch_size,
                train_data,
                sequence_length,
                clip,
                step_vis,
                epoch,
                learning_rate)
        torch.save(model.state_dict(), './snapchot.pth')
        generate(model, 
            words, 
            temperature, 
            corpus)
except KeyboardInterrupt:
    print('-' * 89)
print('Exiting from training early')
