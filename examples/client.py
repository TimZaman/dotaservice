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


def get_reward(prev_state, state):
    """Get the reward."""

    unit_init = get_hero_unit(prev_state)
    unit = get_hero_unit(state)

    reward =  0

    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)

    reward += (xp - xp_init) * 0.002  # One creep will result in 0.114 reward

    if unit_init.is_alive and unit.is_alive:
        hp_init = unit_init.health / unit_init.health_max
        hp = unit.health / unit.health_max
        reward += (hp - hp_init) * 2.0  # Losing 10% hp will result in -0.1 reward
    if unit_init.is_alive and not unit.is_alive:
        reward += -1.0  # Death is a massive -1 penalty.

    return reward


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.affine1b = nn.Linear(1, 128)

        self.affine2a = nn.Linear(128, 2)
        self.affine2b = nn.Linear(128, 2)

        self.saved_log_probs_a = []
        self.saved_log_probs_b = []
        self.rewards = []

    def forward(self, x1, x2):
        xa = F.relu(self.affine1(x1))
        xb = F.relu(self.affine1b(x2))

        x = xa + xb

        action_scores_a = self.affine2a(x)
        action_scores_b = self.affine2b(x)
        return {'x': F.softmax(action_scores_a, dim=1), 'y': F.softmax(action_scores_b, dim=1)}


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)  # 1e-2 seemed fine
eps = np.finfo(np.float32).eps.item()

# pretrained_model = 'runs/Dec0_00-03-40_Tims-Mac-Pro.local/model_000000348.pt'
# pretrained_model = 'runs/Dec05_00/model_000000014.pt'
# policy.load_state_dict(torch.load(pretrained_model), strict=False)
pretrained_model = 'runs/Dec07_15-27-01_ngvpn01-160-168.dyn.scz.us.nvidia.com/model_000000042.pt'
policy.load_state_dict(torch.load(pretrained_model), strict=True)


def select_action(world_state):
    # Preprocess the state
    unit = get_hero_unit(world_state)

    # Location Input
    location_state = np.array([unit.location.x, unit.location.y]) / 7000  # maps the map between [-1 and 1]
    location_state = torch.from_numpy(location_state).float().unsqueeze(0)

    # Health Input
    health_state = torch.from_numpy(np.array([unit.health / unit.health_max])).float().unsqueeze(0) - 1.0 # Map between [-1 and 0]
    
    # TODO(tzaman) add dotatime

    probs = policy(x1=location_state, x2=health_state)

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
    return None



N_RESET_RETRIES = 4

async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 13337, loop=loop)
    env = DotaServiceStub(channel)

    N_STEPS = 10000

    config = Config(
        ticks_per_observation=30,
        host_timescale=10,
        render=False,
        # host_timescale=1,
        # render=True,
    )

    batch_size = 4

    for episode in range(1000):

        all_discounted_rewards = []  # rewards
        actions_steps = []
        reward_sum = 0
        for _ in range(batch_size):
            rewards = []

            state = None
            for i in range(N_RESET_RETRIES):
                try:
                    state = await asyncio.wait_for(env.reset(config), timeout=60)
                    break
                except Exception as e:
                    print('Exception on env.reset: {}'.format(e))
                    if i == N_RESET_RETRIES-1:
                        raise

            for t in range(N_STEPS):  # take 100 steps
                prev_state = state
                action_a, action_b = select_action(state)

                # print('action_a={} action_b={}'.format(action_a, action_b))

                action = CMsgBotWorldState.Action()
                action.actionType = CMsgBotWorldState.Action.Type.Value(
                    'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
                m = CMsgBotWorldState.Action.MoveToLocation()
                hero_unit = get_hero_unit(state)
                hero_location = hero_unit.location
                # print('hero loc x={}, y={}'.format(hero_location.x, hero_location.y))
                m.location.x = hero_location.x + 300 if action_a else hero_location.x - 300
                m.location.y = hero_location.y + 300 if action_b else hero_location.y - 300
                m.location.z = 0

                action.moveToLocation.CopyFrom(m)

                try:
                    state = await asyncio.wait_for(env.step(Action(action=action)), timeout=5)
                except Exception as e:
                    print('Exception on env.step: {}'.format(e))
                    break

                
                # Get the reward for hero 0.
                reward = get_reward(prev_state=prev_state, state=state)

                print('x={:.0f}, y={:.0f}, reward={}'.format(hero_location.x, hero_location.y, reward))

                reward_sum += reward
                rewards.append(reward)
            print('last hero loc dotatime={:.2f}, x={:.0f}, y={:.0f}'.format(state.world_state.dota_time, hero_location.x, hero_location.y))


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
        



# async def main():
#     loop = asyncio.get_event_loop()
#     channel = Channel('127.0.0.1', 13337, loop=loop)
#     env = DotaServiceStub(channel)

#     config = Config(
#         host_timescale=10,
#         ticks_per_observation=30,
#         render=False,
#     )

#     nsteps = 10000
#     nepisodes = 10

#     for e in range(nepisodes):
#         observation = await env.reset(config)
#         rewards = get_rewards(observation)

#         for i in range(nsteps):
#             action = CMsgBotWorldState.Action()
#             action.actionType = CMsgBotWorldState.Action.Type.Value(
#                 'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
#             m = CMsgBotWorldState.Action.MoveToLocation()
#             m.location.x = math.sin(observation.world_state.dota_time) * 500 - 1000
#             m.location.y = math.cos(observation.world_state.dota_time) * 500 - 1000
#             m.location.z = 0
#             action.moveToLocation.CopyFrom(m)
#             observation = await env.step(Action(action=action))
#             rewards = get_rewards(observation)

#             print('t={:.2f}, rewards: {}'.format(observation.world_state.dota_time, rewards))


if __name__ == '__main__':
    asyncio.run(main())
