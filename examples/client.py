from collections import Counter
from time import time
import asyncio
import math
import os
import uuid

from grpclib.client import Channel
from google.protobuf.json_format import MessageToDict
from tensorboardX import SummaryWriter

from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_pb2 import Action
from dotaservice.protos.DotaService_pb2 import Config
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import Status


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(7)

GUI_DEBUG = True
N_STEPS = 10 if GUI_DEBUG else 10

if not GUI_DEBUG:
    writer = SummaryWriter()
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


def get_reward(prev_obs, obs):
    """Get the reward."""
    unit_init = get_hero_unit(prev_obs)
    unit = get_hero_unit(obs)

    reward = {'xp': 0, 'hp': 0, 'death': 0, 'dist': 0, 'lh': 0}


    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)

    reward['xp'] = (xp - xp_init) * 0.002  # One creep will result in 0.114 reward

    if unit_init.is_alive and unit.is_alive:
        hp_init = unit_init.health / unit_init.health_max
        hp = unit.health / unit.health_max
        reward['hp'] = (hp - hp_init) * 1.0
    if unit_init.is_alive and not unit.is_alive:
        reward['death'] = - 0.5  # Death should be a big penalty

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Help him get to mid, for minor speed boost
    dist_mid = math.sqrt(unit.location.x**2 + unit.location.y**2)
    reward['dist'] = -(dist_mid / 8000.) * 0.01

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
optimizer = optim.Adam(policy.parameters(), lr=1e-3)  # 1e-2 is obscene, 1e-4 seems slow.
eps = np.finfo(np.float32).eps.item()

START_EPISODE = 3488
MODEL_FILENAME_FMT = "model_%09d.pt"
pretrained_model = 'runs/Dec12_21-41-31_Tims-Mac-Pro.local/' + MODEL_FILENAME_FMT % START_EPISODE
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
        self.host = host
        self.port = port
        self.config = config

    def create_channel(self):
        loop = asyncio.get_event_loop()
        channel = Channel(self.host, self.port, loop=loop)
        env = DotaServiceStub(channel)
        return env

    ENV_RETRY_DELAY = 5

    async def __call__(self):
        obs = None

        while True:
            # Set up a channel.
            env = self.create_channel()

            # Wait for game to boot.
            response = await asyncio.wait_for(env.reset(self.config), timeout=60)
            obs = response.world_state

            if response.status == Status.Value('OK'):
                print("Channel and reset opened.")
                break
            print("Environment reset request (retrying in {}s):\n{}".format(
                self.ENV_RETRY_DELAY, response))
            await asyncio.sleep(self.ENV_RETRY_DELAY)

        rewards = []
        log_probs = []

        for step in range(N_STEPS):  # Steps/actions in the environment
            prev_obs = obs
            action = select_action(obs, step=step)
            print('action:{}'.format(action))

            log_probs.append({'x': action['x']['logprob'],
                              'y': action['y']['logprob'],
                              'enum': action['enum']['logprob'],
                              })

            action_pb = action_to_pb(action=action, state=obs)

            try:
                response = await asyncio.wait_for(env.step(Action(action=action_pb)), timeout=11)
                obs = response.world_state
            except Exception as e:
                print('Exception on env.step: {}'.format(e))
                break

            reward = get_reward(prev_obs=prev_obs, obs=obs)

            print('{} step={} reward={:.3f}\n'.format(self.port, step, sum(reward.values())))

            rewards.append(reward)

        await env.clear(Empty())

        reward_sum = sum([sum(r.values()) for r in rewards])

        print('{} last dotatime={:.2f}, reward sum={:.2f}'.format(
            self.port, obs.dota_time, reward_sum))

        return rewards, log_probs


def discount_rewards(rewards, gamma=0.99):
    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

async def main():
    loop = asyncio.get_event_loop()



    if GUI_DEBUG:
        config = Config(
            ticks_per_observation=30,
            host_timescale=2,
            render=False,
        )
        n_actors = 1
        # actors = [Actor(config=config, port=13337)]
    else:
        config = Config(
            ticks_per_observation=30,
            host_timescale=10,
            render=False,
        )
        n_actors = 2
        # actors = [
        #     Actor(config=config, port=40000),
        #     Actor(config=config, port=40001),
        #     Actor(config=config, port=40002),
        #     Actor(config=config, port=40003),
        #     Actor(config=config, port=40004),
        #     Actor(config=config, port=40005),
        #     Actor(config=config, port=40006),
        #     Actor(config=config, port=40007),
        # ]

    actors = [Actor(config=config, port=13337) for _ in range(n_actors)]

    N_EPISODES = 100000
    batch_size = 2

    if batch_size % len(actors) != 0:
        raise ValueError('Notice: amount of actors not cleanly divisible by batch size.')

    if n_actors > batch_size:
        raise ValueError('More actors than batch size!')

    for episode in range(START_EPISODE, N_EPISODES):

        all_rewards = []
        all_discounted_rewards = []
        all_log_probs = []

        i = 0
        while i < batch_size:
            print('Sub-batch processed {}/{}'.format(i, batch_size))
            actor_output = await asyncio.gather(*[a() for a in actors])

            # Loop over all distributed actors.
            for rewards, log_probs in actor_output:
                all_rewards.append(rewards)
                combined_rewards = [sum(r.values()) for r in rewards]
                discounted_rewards = discount_rewards(combined_rewards)
                all_discounted_rewards.extend(discounted_rewards)
                all_log_probs.extend(log_probs)
                i += 1

        loss = finish_episode(rewards=all_discounted_rewards, log_probs=all_log_probs)


        

        reward_counter = Counter()
        for b in all_rewards: # jobs in a batch
            for s in b: # Steps in a batch
                reward_counter.update(s)
        reward_counter = dict(reward_counter)
                
        reward_sum = sum(reward_counter.values())
        avg_reward = reward_sum / batch_size
        print('ep={} n_actors={} avg_reward={} loss={}'.format(episode, i, avg_reward, loss))

        if not GUI_DEBUG:
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('mean_reward', avg_reward, episode)
            for k, v in reward_counter.items():
                writer.add_scalar('reward_{}'.format(k), v / batch_size, episode)
            filename = os.path.join(log_dir, MODEL_FILENAME_FMT % episode)
            torch.save(policy.state_dict(), filename)
        

if __name__ == '__main__':
    asyncio.run(main())
