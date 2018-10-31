import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import string

torch.manual_seed(7)

# all_letters = string.ascii_letters + " .,;'"
max_range = 9
all_letters = '0123456789'
n_letters = len(all_letters)
all_categories = list(map(str, range(max_range+1)))
print('all_categories:', all_categories)
n_categories = len(all_categories)

def get_example():
    rmax = np.random.randint(low=0, high=max_range+1)
    rmin = np.random.randint(low=0, high=rmax+1)
    instance = ''.join(map(str, range(rmin, rmax)))
    label = np.random.randint(low=0, high=max_range+1)
    instance = str(label) + instance
    return instance, label


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# print(letterToTensor('J'))

# print(lineToTensor('Jones').size())



def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=1, dropout=0.05)
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        x = self.inp(x.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = self.out(output.squeeze(1))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


n_hidden = 128

model_obj = SimpleRNN
# model_obj = RNN

model = model_obj(input_size=n_letters, hidden_size=n_hidden, output_size=n_categories)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def train(line_tensor, category_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    optimizer.zero_grad()
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()


for i in range(2000):
    instance, label = get_example()
    o, l = train(line_tensor=lineToTensor(instance), category_tensor=torch.tensor([label]))
    cat, cat_i = categoryFromOutput(o)
    print('loss=%.2f instance=%s label=%s pred=%s' % (l, instance, label, cat))


def evaluate(line_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden=hidden)
    return output

output = evaluate(lineToTensor('0'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('34'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('67'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('12345678'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('12567'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('6894517'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('654'))
print(categoryFromOutput(output))

output = evaluate(lineToTensor('123456789'))
print(categoryFromOutput(output))