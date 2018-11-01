import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import string
from collections import namedtuple
from math import hypot

torch.manual_seed(7)

Example = namedtuple('Example', ['depth', 'key_enum', 'value'])

Action = namedtuple('Action', ['selection', 'move_x', 'move_y'])

class DictDataset():
    max_value = 100
    def next(self):
        n_samples = np.random.randint(low=0, high=4)
        x = np.random.uniform(low=0, high=self.max_value)
        y = np.random.uniform(low=0, high=self.max_value)
        instance = {'x': x, 'y': y}
        distance = hypot(x, y)
        selection = distance > self.max_value/3.  # boolean to move y/n
        label = Action(selection=selection, move_x=x if selection else 0, move_y=y if selection else 0)
        return instance, label


class DictExampleDataset():
    keys = ['x', 'y']  # TODO(tzaman): get from proto
    input_size = len(keys) + 1 + 1
    output_size = 1 + 1 + 1
    def __init__(self, dict_dataset):
        self.dict_dataset = dict_dataset

    @classmethod
    def label_to_tensor(self, label):
        tensor = torch.zeros(self.output_size)
        tensor[0] = float(label.selection)
        tensor[1] = label.move_x
        tensor[2] = label.move_y
        return tensor.unsqueeze(0)

    @classmethod
    def example_to_tensor(self, example):
        tensor = torch.zeros(self.input_size)
        tensor[example.key_enum] = 1
        tensor[-2] = example.depth
        tensor[-1] = example.value
        return tensor.unsqueeze(0)

    @classmethod
    def dict_to_examples(self, x):
        examples = []
        for item in x:
            depth = 0
            key_enum = self.keys.index(item)
            value = x[item]
            e = Example(depth=depth, key_enum=key_enum, value=value)
            examples.append(e)
        return examples

    def next(self):
        d, label = self.dict_dataset.next()
        examples = self.dict_to_examples(d)
        return examples, label


dataset = DictExampleDataset(dict_dataset=DictDataset())

print(dataset.next())
print(dataset.next())
print(dataset.next())


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.inp = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=0.05)
        self.out1 = nn.Linear(in_features=hidden_size, out_features=output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        x = self.inp(x.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = self.out1(output.squeeze(1))
        # output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


n_hidden = 128

model_obj = SimpleRNN

output_size = dataset.output_size

model = model_obj(input_size=dataset.input_size, hidden_size=n_hidden, output_size=output_size)
# criterion = nn.NLLLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=1e-5)


def train(instances, label):
    hidden = None
    for ex in instances:
        x = dataset.example_to_tensor(ex)
        output, hidden = model(x, hidden)

    optimizer.zero_grad()
    loss = criterion(output, dataset.label_to_tensor(label))
    loss.backward()
    optimizer.step()

    return output, loss.item()


for i in range(100000):
    instances, label = dataset.next()
    print('instances=%s label=%s' % (instances, label))
    o, l = train(instances=instances, label=label)
    print('  loss=%.2f pred=%s' % (l, o))


def evaluate(line_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden=hidden)
    return output


# output = evaluate(lineToTensor("?"))
# print(categoryFromOutput(output))
