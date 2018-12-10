import asyncio
from time import time
import math
import uuid
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

torch.manual_seed(7)

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
        reward += (hp - hp_init) * 0.4
    if unit_init.is_alive and not unit.is_alive:
        reward += -0.2  # Death should be a big penalty

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward += lh * 0.5

    # Help him get to mid, for minor speed boost
    dist_mid = math.sqrt(unit.location.x**2 + unit.location.y**2)
    reward += (1-(dist_mid / 8000.)) * 0.01

    return reward


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1a = nn.Linear(2, 128)
        self.affine1b = nn.Linear(2, 128)
        self.affine1c = nn.Linear(2, 128)

        self.affine2a = nn.Linear(128, 2)
        self.affine2b = nn.Linear(128, 2)
        self.affine2c = nn.Linear(128, 3)

    def forward(self, xa, xb, xc):
        print('policy(xa={}, xb={}, xc={})'.format(xa, xb, xc))

        xa = self.affine1a(xa)
        xb = self.affine1b(xb)
        xc = self.affine1c(xc)

        x = F.relu(xa + xb + xc)

        action_scores_x = self.affine2a(x)
        action_scores_y = self.affine2b(x)
        action_scores_enum = self.affine2c(x)

        return {
                'x': F.softmax(action_scores_x, dim=1),
                'y': F.softmax(action_scores_y, dim=1),
                'enum': F.softmax(action_scores_enum, dim=1),
                }


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)  # 1e-2 is obscene
eps = np.finfo(np.float32).eps.item()

START_EPISODE = 1177
MODEL_FILENAME_FMT = "model_%09d.pt"
pretrained_model = 'runs/Dec09_15-47-51_Tims-Mac-Pro/' + MODEL_FILENAME_FMT % START_EPISODE
policy.load_state_dict(torch.load(pretrained_model), strict=True)


def select_action(world_state, step=None):
    actions = {}

    # Preprocess the state
    unit = get_hero_unit(world_state)

    # Location Input
    location_state = np.array([unit.location.x, unit.location.y]) / 7000.  # maps the map between [-1 and 1]
    location_state = torch.from_numpy(location_state).float().unsqueeze(0)

    # Health and dotatime input
    hp_rel = 1. - (unit.health / unit.health_max) # Map between [0 and 1]
    dota_time_norm = dota_time = world_state.dota_time / 1200.  # Normalize by 20 minutes
    env_state = torch.from_numpy(np.array([hp_rel, dota_time_norm])).float().unsqueeze(0) 


    # Nearest creep input
    closest_unit, distance = get_nearest_creep_to_hero(world_state, hero_unit=unit)
    MAX_CREEP_DIST = 1200.
    if closest_unit is not None and distance < MAX_CREEP_DIST:
        # print('closest_unit:\n{}'.format(closest_unit))
        creep_hp = 1. - (closest_unit.health / closest_unit.health_max)  # [1 (dead) : 0 (full hp)]
        distance = 1. - (distance / MAX_CREEP_DIST)  # [1 (close): 0 (far)] 
        actions['DOTA_UNIT_ORDER_ATTACK_TARGET'] = {}
        actions['DOTA_UNIT_ORDER_ATTACK_TARGET']['handle'] = closest_unit.handle
    else :
        creep_hp = 0
        distance = 0

    creep_state = torch.from_numpy(np.array([creep_hp, distance])).float().unsqueeze(0) 

    probs = policy(xa=location_state, xb=env_state, xc=creep_state)

    for k, prob in probs.items():
        m = Categorical(prob)
        action = m.sample()
        actions[k] = {'action': action.item(), 'prob': prob, 'logprob' :m.log_prob(action)}

    return actions



def finish_episode(rewards, log_probs):
    print('@finish_eposide')
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    loss = []
    for log_prob, reward in zip(log_probs, rewards):
        for key in log_prob:
            loss.append(-log_prob[key] * reward)

    optimizer.zero_grad()
    loss = torch.cat(loss).mean()
    loss.backward()
    optimizer.step()
    return loss


def get_hero_unit(state, id=0):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.player_id == id:
            return unit
    raise ValueError("hero {} not found in state:\n{}".format(id, state))


def location_distance(lhs, rhs):
    return math.sqrt( (lhs.x-rhs.x)**2  +  (lhs.y-rhs.y)**2 )

def get_nearest_creep_to_hero(state, hero_unit):
    min_d = None
    closest_unit = None
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP') \
            and unit.team_id != hero_unit.team_id \
            and unit.is_alive:  # Why the shits does the proto show dead units.
            d = location_distance(hero_unit.location, unit.location)
            if min_d is None or d < min_d:
                min_d = d
                closest_unit = unit
    return closest_unit, min_d


N_STEPS = 120
N_RESET_RETRIES = 4


def action_to_pb(action, state):
    hero_unit = get_hero_unit(state)

    action_pb = CMsgBotWorldState.Action()
    action_enum = action['enum']['action']
    if action_enum == 0:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    elif action_enum == 1:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
            'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
        m = CMsgBotWorldState.Action.MoveToLocation()
        hero_location = hero_unit.location
        m.location.x = hero_location.x + 350 if action['x']['action'] else hero_location.x - 350
        m.location.y = hero_location.y + 350 if action['y']['action'] else hero_location.y - 350
        m.location.z = 0
        action_pb.moveToLocation.CopyFrom(m)
    elif action_enum == 2:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                            'DOTA_UNIT_ORDER_ATTACK_TARGET')
        m = CMsgBotWorldState.Action.AttackTarget()
        if 'DOTA_UNIT_ORDER_ATTACK_TARGET' in action:
            m.target = action['DOTA_UNIT_ORDER_ATTACK_TARGET']['handle']
        else:
            m.target = -1
        m.once = False
        action_pb.attackTarget.CopyFrom(m)
    else:
        raise ValueError("unknown action {}".format(action_enum))

    return action_pb

class Actor(object):

    def __init__(self, config, host='127.0.0.1', port=13337):
        loop = asyncio.get_event_loop()
        channel = Channel(host, port, loop=loop)
        self.host = host
        self.port = port
        self.env = DotaServiceStub(channel)
        self.config = config

    async def __call__(self):
        rewards = []
        state = None
        for i in range(N_RESET_RETRIES):
            try:
                state = await asyncio.wait_for(self.env.reset(self.config), timeout=60)
                break
            except Exception as e:
                print('Exception on env.reset: {}'.format(e))
                if i == N_RESET_RETRIES-1:
                    raise

        log_probs = []

        for step in range(N_STEPS):  # Steps/actions in the environment
            prev_state = state
            action = select_action(state.world_state, step=step)
            print('action:{}'.format(action))

            log_probs.append({'x': action['x']['logprob'],
                              'y': action['y']['logprob'],
                              'enum': action['enum']['logprob'],
                              })

            action_pb = action_to_pb(action=action, state=state.world_state)

            try:
                state = await asyncio.wait_for(self.env.step(Action(action=action_pb)), timeout=11)
            except Exception as e:
                print('Exception on env.step: {}'.format(e))
                break

            reward = get_reward(prev_state=prev_state.world_state, state=state.world_state)

            print('{} step={} reward={:.3f}\n'.format(self.port, step, reward))

            rewards.append(reward)

        print('{} last dotatime={:.2f}, reward sum={:.2f}'.format(
            self.port, state.world_state.dota_time, sum(rewards)))

        return rewards, log_probs


def discount_rewards(rewards, gamma=0.99):
    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

GUI_DEBUG = False

async def main():
    loop = asyncio.get_event_loop()

    if GUI_DEBUG:
        config = Config(
            ticks_per_observation=30,
            host_timescale=10,
            render=False,
        )
        actors = [Actor(config=config, port=13337)]
    else:
        config = Config(
            ticks_per_observation=30,
            host_timescale=10,
            render=False,
        )
        actors = [
            Actor(config=config, port=42000),
            Actor(config=config, port=42001),
            Actor(config=config, port=42002),
            Actor(config=config, port=42003),
            Actor(config=config, port=42004),
            Actor(config=config, port=42005),
            Actor(config=config, port=42006),
            Actor(config=config, port=42007),
        ]

    N_EPISODES = 100000
    BATCH_SIZE = 8

    if BATCH_SIZE % len(actors) != 0:
        print('Notice: amount of actors not cleanly divisible by batch size.')

    for episode in range(START_EPISODE, N_EPISODES):

        reward_sum = 0
        all_rewards = []
        all_log_probs = []

        i = 0
        while i < BATCH_SIZE:
            print('@subbatch #{}'.format(i))
            actor_output = await asyncio.gather(*[a() for a in actors])

            # Loop over all distributed actors.
            for rewards, log_probs in actor_output:
                reward_sum += sum(rewards)
                discounted_rewards = discount_rewards(rewards)
                all_rewards.extend(discounted_rewards)
                all_log_probs.extend(log_probs)

                i += 1

        loss = finish_episode(rewards=all_rewards, log_probs=all_log_probs)

        avg_reward = reward_sum / BATCH_SIZE
        print('ep={} n_actors={} avg_reward={} loss={}'.format(episode, i, avg_reward, loss))
        writer.add_scalar('loss', loss, episode)
        writer.add_scalar('mean_reward', avg_reward, episode)
        filename = os.path.join(log_dir, MODEL_FILENAME_FMT % episode)
        torch.save(policy.state_dict(), filename)
        

if __name__ == '__main__':
    asyncio.run(main())
