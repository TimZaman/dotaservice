import asyncio
from time import time
import math
import uuid
# from itertools import count
import os

from grpclib.client import Channel
from google.protobuf.json_format import MessageToDict
from tensorboardX import SummaryWriter

from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_pb2 import Action
from dotaservice.protos.DotaService_pb2 import Config

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(1337)

writer = SummaryWriter()
if writer:
    log_dir = writer.file_writer.get_logdir()

xp_to_reach_level = {
    1: 0,
    2: 230,
    3: 600,
    4: 1080,
    5: 1680,
    6: 2300,
    7: 2940,
    8: 3600,
    9: 4280,
    10: 5080,
    11: 5900,
    12: 6740,
    13: 7640,
    14: 8865,
    15: 10115,
    16: 11390,
    17: 12690,
    18: 14015,
    19: 15415,
    20: 16905,
    21: 18405,
    22: 20155,
    23: 22155,
    24: 24405,
    25: 26905
}


def get_total_xp(level, xp_needed_to_level):
    if level == 25:
        return xp_to_reach_level[level]
    xp_required_for_next_level = xp_to_reach_level[level + 1] - xp_to_reach_level[level]
    missing_xp_for_next_level = (xp_required_for_next_level - xp_needed_to_level)
    return xp_to_reach_level[level] + missing_xp_for_next_level


def get_rewards(observation):
    """Get the rewards for all heroes."""
    rewards = {}
    for unit in observation.world_state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
            reward = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)
            rewards[unit.player_id] = reward
    return rewards


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.affine2a = nn.Linear(128, 2)
        self.affine2b = nn.Linear(128, 2)

        self.saved_log_probs_a = []
        self.saved_log_probs_b = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores_a = self.affine2a(x)
        action_scores_b = self.affine2b(x)
        return {'x': F.softmax(action_scores_a, dim=1), 'y': F.softmax(action_scores_b, dim=1)}


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

# pretrained_model = 'model_000000014.pt'
# policy.load_state_dict(torch.load(pretrained_model))


def select_action(state):
    # Preprocess the state
    state = get_hero_unit(state).location
    state = np.array([state.x, state.y]) / 7000  # maps the map between [-1 and 1]

    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)

    m = Categorical(probs['x'])
    action_a = m.sample()
    policy.saved_log_probs_a.append(m.log_prob(action_a))

    m = Categorical(probs['y'])
    action_b = m.sample()
    policy.saved_log_probs_b.append(m.log_prob(action_b))

    return action_a.item(), action_b.item()


def finish_episode():
    policy_loss = []
    rewards = policy.rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for log_prob, reward in zip(policy.saved_log_probs_a, rewards):
        policy_loss.append(-log_prob * reward)

    for log_prob, reward in zip(policy.saved_log_probs_b, rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs_a[:]
    del policy.saved_log_probs_b[:]


def get_hero_unit(state, id=0):
   for unit in state.world_state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.player_id == id:
            return unit


async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 13337, loop=loop)
    env = DotaServiceStub(channel)

    config = Config(
        host_timescale=10,
        ticks_per_observation=30,
        render=False,
        # render=True,
    )

    log_interval = 10
    batch_size = 4

    running_reward = 10

    for episode in range(1000):

        all_discounted_rewards = []  # rewards
        actions_steps = []
        reward_sum = 0
        for _ in range(batch_size):
            rewards = []

            state = await env.reset(config)

            for t in range(200):  # take 100 steps
                action_a, action_b = select_action(state)

                # print('action_a={} action_b={}'.format(action_a, action_b))

                action = CMsgBotWorldState.Action()
                action.actionType = CMsgBotWorldState.Action.Type.Value(
                    'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
                m = CMsgBotWorldState.Action.MoveToLocation()
                hero_unit = get_hero_unit(state)
                hero_location = hero_unit.location
                # print('hero loc x={}, y={}'.format(hero_location.x, hero_location.y))
                m.location.x = hero_location.x + 100 if action_a else hero_location.x - 100
                m.location.y = hero_location.y + 100 if action_b else hero_location.y - 100
                m.location.z = 0

                action.moveToLocation.CopyFrom(m)
                state = await env.step(Action(action=action))

                # Get the reward for hero 0.
                reward = get_rewards(state)[0]

                # Factor in health.
                reward *= hero_unit.health / hero_unit.health_max

                reward_sum += reward
                rewards.append(reward)
            print('last hero loc dotatime={}, x={}, y={}'.format(state.world_state.dota_time, hero_location.x, hero_location.y))


            discounted_rewards = []
            R = 0
            gamma = 0.99

            for r in rewards[::-1]:
                R = r + gamma * R
                discounted_rewards.insert(0, R)

            policy.rewards.extend(discounted_rewards)


        finish_episode()

        avg_reward = reward_sum/batch_size
        print('ep={} reward sum={}'.format(episode, reward_sum/batch_size))
        writer.add_scalar('mean_reward', avg_reward, episode)
        filename = os.path.join(log_dir, "model_%09d.pt" % episode)
        torch.save(policy.state_dict(), filename)

if __name__ == '__main__':
    asyncio.run(main())
