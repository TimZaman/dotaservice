import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import string
from recordtype import recordtype
from math import hypot

torch.manual_seed(7)

Example = recordtype('Example', ['depth', 'key_enum', 'value'])

Action = recordtype('Action', ['selection', 'move_x', 'move_y'])

class DictDataset():
    max_value = 80
    def next(self):
        n_samples = np.random.randint(low=0, high=4)
        x = np.random.uniform(low=-self.max_value, high=self.max_value)
        y = np.random.uniform(low=-self.max_value, high=self.max_value)
        instance = {'x': x, 'y': y}
        distance = hypot(x, y)
        selection = distance > self.max_value/3.  # boolean to move y/n
        # label = Action(selection=selection, move_x=x if selection else 0, move_y=y if selection else 0)
        return instance


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
        d = self.dict_dataset.next()
        examples = self.dict_to_examples(d)
        return examples


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
                           num_layers=num_layers)#, dropout=0.05)
        self.out1 = nn.Linear(in_features=hidden_size, out_features=output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        x = self.inp(x.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = self.out1(output.squeeze(1))
        # output = self.softmax(output)
        # print('out=', output)
        output = torch.sigmoid(output)
        # print('sigmoid(out)=', output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


n_hidden = 4

model_obj = SimpleRNN

output_size = dataset.output_size

model = model_obj(input_size=dataset.input_size, hidden_size=n_hidden, output_size=output_size)
# criterion = nn.NLLLoss()
# criterion = nn.MSELoss()
criterion = torch.nn.MSELoss(reduction='none')

# optimizer = optim.Adam(model.parameters(), lr=0.000001)#, weight_decay=1e-5)
optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

def encode_and_policy(instances):
    # TODO(tzaman): separate the encoder and policy into distinct networks.
    hidden = None
    for ex in instances:
        x = dataset.example_to_tensor(ex)
        output, hidden = model(x, hidden)


    # for ex in instances:
    #     x = dataset.example_to_tensor(ex)


    return output

def update_state(instances, action):
    # Notice `instances` is changed in-place
    move = action[0][0] == 1
    move_x = action[0][1] == 1
    move_y = action[0][2] == 1
    # print('instances=', instances)
    # print(move_x)
    # print(move_y)

    # if move:
    if move_x:
        instances[0].value -= 10
    else:
        instances[0].value += 10
    if move_y:
        instances[1].value -= 10
    else:
        instances[1].value += 10
    return instances


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r



def train(instances):
    optimizer.zero_grad()
    # print(instances)
    # for instance in instances:
    #     x = torch.tensor(instance.value, dtype=torch.float32)
    #     print(x)
    #     instance.value = x

    # for param in model.parameters():
    #     print('param:', param)
    #     print(param.data)

    drs = []  # rewards
    # dlogps = []
    action_probs = []
    fake_labels = []
    reward = -1
    # Go through an episode
    for i in range(16):
        # print("(x={x},y={y})".format(x=instances[0].value, y=instances[1].value))        
        
        action_prob = encode_and_policy(instances)
        # print('action_prob:', action_prob)

        # Probe a random action
        action_rand = torch.rand_like(action_prob)

        # "grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        action = action_rand < action_prob.detach()  # Does this detach the gradient tape?
        # print('action:', action)
        fake_labels.append(1 - action)

        action_probs.append(action_prob)

        # Update the state given the action.
        instances = update_state(instances=instances, action=action)
        
        
        if i == 15:
            if hypot(instances[0].value, instances[1].value) < 20:
                drs.append(1)
                # print("                                    nice!")
            else:
                drs.append(-1)
                # print("FAIL")
        else:
            drs.append(0)
            

    action_probs = torch.cat(action_probs).float()
    fake_labels = torch.cat(fake_labels).float()
    
    # print('drs:', drs)
    print('action_probs:', action_probs)
    print('fake_labels:', fake_labels)


    # lx = torch.stack(action_probs).float()
    # ly = torch.tensor(fake_labels).float()
    losses = criterion(action_probs, fake_labels)
    # print('losses:', losses)

    # exit()

    epr = np.array(drs).astype(np.float)
    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    print('discounted_epr', discounted_epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.max([np.std(discounted_epr), 1e-9])


    t_discounted_epr = torch.tensor(discounted_epr, dtype=torch.float32)
    
    t_discounted_epr = t_discounted_epr.expand([3, -1])#.repeat([3,1])
    t_discounted_epr = t_discounted_epr.t()

    print('t_discounted_epr', t_discounted_epr)

    # print('losses', losses)
    losses = torch.mul(t_discounted_epr, losses)
    # print(losses)
    # losses *=  t_discounted_epr                
    loss = torch.mean(losses)  

    # print(loss)
    loss.backward()
    optimizer.step()
    
    # raw_input(' .Press Enter..')

    # exit()

    # return output, loss.item()
    # return 1337, loss.item()
    return drs[-1]

all_rewards = []
for i in range(100000):
    instances = dataset.next()
    # print('instances=%s' % (instances))
    reward = train(instances=instances)
    all_rewards.append(reward)
    print(np.mean(all_rewards[-100:-1]))
    # print('  loss=%.2f pred=%s' % (l, o))


def evaluate(line_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden=hidden)
    return output


# output = evaluate(lineToTensor("?"))
# print(categoryFromOutput(output))