import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import string
from recordtype import recordtype
from math import hypot
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

writer = SummaryWriter()
torch.manual_seed(7)

Example = recordtype('Example', ['depth', 'key_enum', 'value'])

class DictDataset():
    max_value = 80
    def next(self):
        n_samples = np.random.randint(low=0, high=4)
        x = np.random.uniform(low=-self.max_value, high=self.max_value)
        y = np.random.uniform(low=-self.max_value, high=self.max_value)
        instance = {'x': x, 'y': y}
        # distance = hypot(x, y)
        # selection = distance > self.max_value/3.  # boolean to move y/n
        return instance


class DictExampleDataset():
    keys = ['x', 'y']  # TODO(tzaman): get from proto
    # input_size = len(keys) + 1 + 1
    input_size = 1
    output_size = 2# + 1 + 1
    def __init__(self, dict_dataset):
        self.dict_dataset = dict_dataset

    # @classmethod
    # def label_to_tensor(self, label):
    #     tensor = torch.zeros(self.output_size)
    #     tensor[0] = float(label.selection)
    #     tensor[1] = label.move_x
    #     tensor[2] = label.move_y
    #     return tensor.unsqueeze(0)

    @classmethod
    def example_to_tensor(self, example):
        tensor = torch.zeros(self.input_size)
        # tensor[example.key_enum] = 1
        # tensor[-2] = example.depth
        tensor[-1] = example.value
        return tensor#.unsqueeze(0)

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

# print(dataset.next())
# print(dataset.next())
# print(dataset.next())

class Action(object):

    def __init__(self, head, prob):
        self.head = head
        self.prob = prob
        m = Categorical(prob)
        self.sample = m.sample()
        self.logprob = m.log_prob(self.sample)



class Head(object):
    """An action is something that will be done to change the state."""

    def __init__(self, name):
        self.name = name

    def create_head(self):
        self.head = nn.Linear(16, 2)

    def __call__(self, x):
        """Calling the head creates an action, associated with this head."""
        x = self.head(x)
        action_prob = F.softmax(x, dim=1)
        return Action(head=self, prob=action_prob)


class Policy(nn.Module):
    def __init__(self, heads):
        super(Policy, self).__init__()
        self._heads = heads

        self.affine1 = nn.Linear(2, 16)

        for head in self._heads: head.create_head()

    def forward(self, x, hidden=None):
        x = F.relu(self.affine1(x))
        actions = []
        for head in self._heads:
            actions.append(head(x))
        return actions


# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size
#         # print('input_size', input_size)
#         # exit()
#         self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
#         # self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
#         #                    num_layers=num_layers)#, dropout=0.05)
#         self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
#         # self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x, hidden=None):
#         # print('')
#         # print('x=', x)
#         # xhat = 
#         x = self.fc1(x)
#         # print("fc(x)=", x)
#         # x = self.fc1(x)
#         # output, hidden = self.rnn(x, hidden)
#         # output = torch.relu(x)
#         output = F.relu(x)
#         # print("relu(fc1(x'))=", output)
#         # output = x
#         output = self.fc2(output)
#         # print("fc2(relu(fc1(x)))=", output)
#         # output = self.softmax(output)
#         # print('out=', output)
#         # output = torch.sigmoid(output)#.unsqueeze(0)
#         output = F.softmax(output, dim=1)
#         # print('sigmoid(out)=', output)
#         # exit()
#         return output, hidden

#     def init_hidden(self):
#         return torch.zeros(1, 1, self.hidden_size)


n_hidden = 32

# model_obj = SimpleRNN

# output_size = dataset.output_size

# model = model_obj(input_size=dataset.input_size, hidden_size=n_hidden, output_size=output_size)


head_x = Head(name='x')
head_y = Head(name='y')

heads = [head_x, head_y]

model = Policy(heads=heads)

# criterion = nn.NLLLoss()
# criterion = nn.MSELoss()
# criterion = torch.nn.MSELoss(reduction='none')
# criterion = torch.nn.BCELoss(reduction='none')

optimizer = optim.Adam(model.parameters(), lr=1e-2)#, weight_decay=1e-5)
# optimizer = optim.RMSprop(model.parameters(), lr=1e-6)

def encode_and_policy(instances):
    # print('instances=', instances)
    # TODO(tzaman): separate the encoder and policy into distinct networks.
    # hidden = None
    # for ex in instances:
    #     x = dataset.example_to_tensor(ex)
    #     output, hidden = model(x, hidden)


    # xy = []
    # for ex in instances:
        
    #     xy.append(x[0][-1].unsqueeze(0))

    # xy = torch.cat(xy)
    
    # print('xy=', xy)
    xy = [dataset.example_to_tensor(instances[0]), dataset.example_to_tensor(instances[1])]
    xy = torch.cat(xy).unsqueeze(0)
    xy /= 100.  # normalize
    # print('x=', x)
    # output, hidden = model(x, hidden)
    # print('xy=', xy)
    actions = model(xy)
    # exit()


    return actions

def update_state(instances, actions):
    # Notice `instances` is changed in-place
    # print('instances=', instances)
    # print('actions=', actions)
    

    # action = actions[0]
    move_x = actions[0].sample
    move_y = actions[1].sample
    # move_x = actions[0]

    if move_x:
        instances[0].value -= 10
    else:
        instances[0].value += 10
    if move_y:
        instances[1].value += 10
    else:
        instances[1].value -= 10
    return instances


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

batch_size = 4

def train(episode_number, instances):

    # print(instances)
    # for instance in instances:
    #     x = torch.tensor(instance.value, dtype=torch.float32)
    #     print(x)
    #     instance.value = x

    # for param in model.parameters():
    #     print('param:', param)
    #     print(param.data)

    rewards = []  # rewards
    # dlogps = []
    # action_probs = []
    # fake_labels = []
    # log_probs = []
    reward = -1
    # Go through an episode
    # print("(x={x},y={y})".format(x=instances[0].value, y=instances[1].value))
    nsteps = 10
    actions_steps = []
    for i in range(nsteps):
        # print('')
        actions = encode_and_policy(instances)
        
        # print('action_prob:', action_prob)
        # print('actions:', actions)

        # Probe a random action
        # action_rand = torch.rand_like(action_prob)
        # print('action_rand:', action_rand)

        # "grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        # action = action_rand < action_prob.detach()  # Does this detach the gradient tape?

        # print('action', action)

        # fake_label = 1 - action
        # fake_labels.append(fake_label)

        # print('fake_label', fake_label)

        # action_probs.append(action_prob)

        # print('action_prob:', action_prob)
        # m = Categorical(action_prob)
        # action = m.sample()
        # print('Action:', action)
        # logprob = m.log_prob(action)
        # log_probs.append(logprob)
        # print('logprob:', logprob)
        # print('action.item()', action.item())

        # Update the state given the action.
        instances = update_state(instances=instances, actions=actions)
        actions_steps.append(actions)
        
        if i == nsteps-1:
            if hypot(instances[0].value, instances[1].value) < 20:
                # if abs(instances[0].value) < 15:
                rewards.append(1)
                # raw_input("Noise! Press Enter..")
                # print("                                    nice!")
            else:
                rewards.append(-1)
                # print("FAIL")
        else:
            rewards.append(0)
        # print("(x={x},y={y})".format(x=instances[0].value, y=instances[1].value))    
    # print('actions_steps', actions_steps)
    # print('AP', action_probs)
    # action_probs = torch.stack([a.prob for a in actions_steps]).float()
    # fake_labels = torch.cat(fake_labels).float()
    
    # print('rewards:', rewards)
    # print('action_probs:', action_probs)
    # print('fake_labels:', fake_labels)

  

    
    # log_probs = torch.stack(log_probs)
    
    # exit()


    # exit()

    epr = np.array(rewards).astype(np.float)
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.max([np.std(discounted_epr), 1e-9])
    t_discounted_epr = torch.tensor(discounted_epr, dtype=torch.float32)#.unsqueeze(0)
    # print('t_discounted_epr', t_discounted_epr)


    # Get the losses for each action.

    logprobs = {head_x: [], head_y: []}
    for action_step in actions_steps:
        for action in action_step:
            logprobs[action.head].append(action.logprob)

    head_losses = []
    for logprob_head in logprobs.values():
        # print('hi head')
        # print(logprobs)
        # print()
        # print(torch.stack(logprob_head))
        # xxx = 
        # print('xxx=', xxx)
        # print('t_discounted_epr=', t_discounted_epr)
        head_loss = torch.mul(-torch.cat(logprob_head), t_discounted_epr)
        #exit()#
        # print('headloss=', head_loss)
        head_losses.append(head_loss)
        # head_losses = torch.cat([head_losses, head_loss])
    head_losses = torch.stack(head_losses)

    # print('head_losses', head_losses)
    # exit()

    # losses = -log_probs * t_discounted_epr

    loss = torch.sum(head_losses)  
    # print('loss=', loss)



    loss.backward()
    if episode_number % batch_size == 0 and episode_number > 0:                   
        optimizer.step()
        optimizer.zero_grad()       

    return rewards[-1]

all_rewards = []
for i in range(1000000):
    instances = dataset.next()
    # print('instances=%s' % (instances))
    reward = train(episode_number=i, instances=instances)
    all_rewards.append(reward)
    running_avg_reward = np.mean(all_rewards[-200:-1])
    print('%.3f' % running_avg_reward)
    writer.add_scalar('reward', reward, i)
    # print('  loss=%.2f pred=%s' % (l, o))


def evaluate(line_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden=hidden)
    return output


# output = evaluate(lineToTensor("?"))
# print(categoryFromOutput(output))