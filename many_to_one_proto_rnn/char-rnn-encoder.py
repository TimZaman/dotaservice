import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import string
from collections import namedtuple

torch.manual_seed(7)

Example = namedtuple('Example', ['depth', 'key_enum', 'value'])


class DictDataset():
    max_range = 9
    def next(self):
        n_samples = np.random.randint(low=0, high=4)
        instance = {}
        label1 = np.random.randint(low=0, high=self.max_range+1)
        label2 = np.random.randint(low=0, high=self.max_range - label1 + 1)
        label = label1 + label2
        instance['invoker'] = label1
        instance['zeus'] = label2
        return instance, label


class DictExampleDataset():
    input_size = 3
    keys = ['zeus', 'invoker']
    def __init__(self, dict_dataset):
        self.dict_dataset = dict_dataset

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

output_size = 1

model = model_obj(input_size=dataset.input_size, hidden_size=n_hidden, output_size=output_size)
# criterion = nn.NLLLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=1e-5)

def train(instances, category_tensor):
    hidden = None
    for ex in instances:
        x = torch.tensor([ex.depth, ex.key_enum, ex.value], dtype=torch.float)
        output, hidden = model(x, hidden)

    optimizer.zero_grad()
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()


for i in range(100000):
    instances, label = dataset.next()
    print(instances)
    print(label)
    o, l = train(instances=instances, category_tensor=torch.tensor([[label]], dtype=torch.float))
    # cat, cat_i = categoryFromOutput(o)
    print('loss=%.2f instances=%s label=%s pred=%s' % (l, instances, label, o))


def evaluate(line_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden=hidden)
    return output


output = evaluate(lineToTensor("{'foo': 3, 'invoker': 5, 'bar': 2}'}"))
print(categoryFromOutput(output))

# output = evaluate(lineToTensor('0'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('34'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('67'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('12345678'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('12567'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('6894517'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('654'))
# print(categoryFromOutput(output))

# output = evaluate(lineToTensor('123456789'))
# print(categoryFromOutput(output))