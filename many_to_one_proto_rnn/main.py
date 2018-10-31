import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import string

torch.manual_seed(7)

# all_letters = string.ascii_letters + " .,;'"
max_range = 9
all_letters = '0123456789'
n_letters = len(all_letters)
all_categories = list(map(str, range(max_range+1)))
print('all_categories:', all_categories)

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

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_categories = 10


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()

# input = lineToTensor('Albert')
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input[0], hidden)
# print(output)


# learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.001

def train(category_tensor, line_tensor):
    # print(category_tensor)
    # print(line_tensor)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

for i in range(100000):
    rmax = np.random.randint(low=1, high=max_range+1)
    rmin = np.random.randint(low=0, high=rmax)
    line = ''.join(map(str, range(rmin, rmax)))
    # target = rmax
    # target = rmin
    target = np.random.randint(low=1, high=max_range+1)
    line = str(target) + line
    o, l = train(category_tensor=torch.tensor([target]), line_tensor=lineToTensor(line))

    print('loss=%.2f line=%s target=%s' % (l, line, target))


# hidden = rnn.initHidden()
# for c in '0123':
#     o, hidden = rnn(letterToTensor(c), hidden)
#     print(o)
print('---')
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

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

output = evaluate(lineToTensor('123456789'))
print(categoryFromOutput(output))