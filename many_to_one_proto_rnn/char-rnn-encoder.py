import os

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
# writer = None
if writer:
    log_dir = writer.file_writer.get_logdir()
torch.manual_seed(7)

pretrained_model = None
# pretrained_model = "/Users/tzaman/Drive/code/dotabot/many_to_one_proto_rnn/runs/Nov10_12-33-14_Tims-Mac-Pro.local/model_001360000_l0.63.pt"

# print(torch.load(pretrained_model))
# exit()

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
    input_size = len(keys) + 1 + 1
    # input_size = 1
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
        tensor[example.key_enum] = 1 - 0.5
        tensor[-2] = example.depth
        tensor[-1] = example.value / 100. # normalize
        # print('tensor', tensor)
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
        m = Categorical(self.prob)
        self.sample = m.sample()
        self.logprob = m.log_prob(self.sample)
        # self.logprob = -(1 - self.prob[self.sample])
        # print('logprob:', self.logprob)



class Head(object):
    """An action is something that will be done to change the state."""

    def __init__(self, name):
        self.name = name

    def create_head(self):
        self.head = nn.Linear(16, 2)

    def __call__(self, x):
        """Calling the head creates an action, associated with this head."""
        x = self.head(x)
        x = x.view(-1)
        # print('head(x)=', x)
        action_prob = F.softmax(x, dim=-1)
        # print('softmax(x)=', action_prob)

        return Action(head=self, prob=action_prob)


class Policy(nn.Module):
    def __init__(self, heads):
        super(Policy, self).__init__()
        self._heads = heads

        self.fc1 = nn.Linear(4, 16)

        self.rnn = nn.LSTM(input_size=16, hidden_size=16,
                           num_layers=1)#, dropout=0.05)

        for head in self._heads: head.create_head()

        self.head1 = self._heads[0].head  # HACK
        self.head2 = self._heads[1].head  # HACK

    def forward_rnn(self, x, hidden=None):
        # print()
        # print('x=', x)
        x = x.view(1, -1)
        # print('x=', x)
        x = F.relu(self.fc1(x)).unsqueeze(1)
        # print('relu(fc(x))=', x)
        x, hidden = self.rnn(x, hidden)
        return x, hidden

    def forward_mlp(self, x):
        # print('x =', x)
        # x = x.view(1, -1)
        # print('x =', x)
        x = F.relu(self.fc1(x))#.unsqueeze(1)
        # print('relu(fc(x))=', x)

        return x

    def forward_heads(self, x):

        # x = F.relu(x)
        # print('rnn()=', x)
        actions = []
        for head in self._heads:
            actions.append(head(x))
        # raw_input('Enter..')
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



head_x = Head(name='x')
head_y = Head(name='y')

heads = [head_x, head_y]

model = Policy(heads=heads)
print(model)
# criterion = nn.NLLLoss()
# criterion = nn.MSELoss()
# criterion = torch.nn.MSELoss(reduction='none')
# criterion = torch.nn.BCELoss(reduction='none')

optimizer = optim.Adam(model.parameters(), lr=1e-2)#, weight_decay=1e-5)
# optimizer = optim.RMSprop(model.parameters(), lr=1)

def encode_and_policy(instances):
    # print('instances=', instances)
    # TODO(tzaman): separate the encoder and policy into distinct networks.
    hidden = None
    for example in instances:
        example_tensor = dataset.example_to_tensor(example)
        output, hidden = model.forward_rnn(example_tensor, hidden=hidden)
    

    # example_tensor = torch.stack([dataset.example_to_tensor(instances[0])[-1], dataset.example_to_tensor(instances[1])[-1]])
    # # print('example_tensor', example_tensor)
    # output = model.forward_mlp(example_tensor)
    


    actions = model.forward_heads(output)

    # print('actions=', [actions[0].prob, actions[0].prob])

    return actions

def update_state(instances, actions):
    # Notice `instances` is changed in-place
    # print('instances=', instances)
    # print('actions=', actions)

    move_x = actions[0].sample
    move_y = actions[1].sample
    # print('move_x={}, move_y={}'.format(move_x, move_y))
    if move_x:
        instances[0].value += 10
    else:
        instances[0].value -= 10
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

batch_size = 32

def train(episode_number, dataset):

    all_discounted_rewards = []  # rewards
    actions_steps = []
    reward_sum = 0

    for _ in range(batch_size):
        instances = dataset.next()
        rewards = []
        reward = -1
        nsteps = 10
        # print('')
        # print("(x={x},y={y})".format(x=instances[0].value, y=instances[1].value))    
        for i in range(nsteps):
            # print('')
            actions = encode_and_policy(instances)
            # print(' prob', [a.prob for a in actions])
            # print(' sample', [a.sample for a in actions])
            # print(' logprob', [a.logprob for a in actions])
            # Update the state given the action.
            instances = update_state(instances=instances, actions=actions)
            actions_steps.append(actions)
            
            if i == nsteps-1:
                if hypot(instances[0].value, instances[1].value) < 20:
                    rewards.append(1)
                else:
                    rewards.append(-1)
            else:
                rewards.append(0)
            reward_sum += rewards[-1]
            # print(instances)
            # rewards.append(-hypot(instances[0].value, instances[1].value)/100. + 0.5)
            # print("(x={x},y={y})".format(x=instances[0].value, y=instances[1].value))    

        # Discount the rewards
        discounted_rewards = []
        R = 0
        gamma = 0.99
        eps = 1e-7
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        all_discounted_rewards.extend(discounted_rewards)

    # print('rewards:', rewards)
    # if rewards[-1] != 1:
    # input('Enter')

    # epr = np.array(rewards).astype(np.float)
    # discounted_epr = discount_rewards(epr)
    # discounted_epr -= np.mean(discounted_epr)
    # discounted_epr /= np.max([np.std(discounted_epr), 1e-7])
    # t_discounted_epr = torch.tensor(discounted_epr, dtype=torch.float32)#.unsqueeze(0)
    # print('t_discounted_epr', t_discounted_epr)


    # rewards_norm = []
    # R = 0
    # gamma = 0.99
    # eps = 1e-7
    # for r in rewards[::-1]:
    #     R = r + gamma * R
    #     rewards_norm.insert(0, R)
    # print('all_discounted_rewards:', all_discounted_rewards)
    rewards_norm = torch.tensor(all_discounted_rewards)
    rewards_norm = (rewards_norm - rewards_norm.mean()) / (rewards_norm.std() + eps)

    # print('rewards_norm:', rewards_norm)

    t_discounted_epr = rewards_norm


    # Get the losses for each action.

    logprobs = {head_x: [], head_y: []}
    for action_step in actions_steps:
        for action in action_step:
            logprobs[action.head].append(action.logprob)

    head_losses = []
    for logprob_head in logprobs.values():
        head_loss = -torch.stack(logprob_head) * t_discounted_epr
        head_losses.append(head_loss)
    head_losses = torch.stack(head_losses)

    loss = torch.sum(head_losses)  

    # if episode_number % batch_size == 0 and episode_number > 0:                
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()       

    return reward_sum / batch_size

if pretrained_model:
    model.load_state_dict(torch.load(pretrained_model))


SUMMARY_EVER_N_STEPS = 1

all_rewards = []
for i in range(100000000):
    # print('instances=%s' % (instances))
    reward = train(episode_number=i, dataset=dataset)
    # all_rewards.append(reward)
    # running_avg_reward = np.mean(all_rewards[-200:-1])
    if i % SUMMARY_EVER_N_STEPS == 0:
        print('%.3f' % reward)
        if writer:
            writer.add_scalar('reward', reward, i)
        # print('  loss=%.2f pred=%s' % (l, o))

    if writer and i % 10000 == 0:
        filename = os.path.join(log_dir, "model_%09d_l%.2f.pt" % (i, reward))
        torch.save(model.state_dict(), filename)


# def evaluate(line_tensor):
#     hidden = None
#     for i in range(line_tensor.size()[0]):
#         output, hidden = model(line_tensor[i], hidden=hidden)
#     return output


# # output = evaluate(lineToTensor("?"))
# # print(categoryFromOutput(output))