from collections import Counter
from time import time
import asyncio
import logging
import math
import os
import time
import traceback
import uuid

from google.cloud import storage
from google.protobuf.json_format import MessageToDict
from grpclib.client import Channel
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Action
from dotaservice.protos.DotaService_pb2 import Config
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import Status

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

torch.manual_seed(7)

client = storage.Client()
bucket = client.get_bucket('dotaservice')

USE_CHECKPOINTS = False
N_STEPS = 150
START_EPISODE = 5491
MODEL_FILENAME_FMT = "model_%09d.pt"
TICKS_PER_OBSERVATION = 30
N_DELAY_ENUMS = 5
HOST_TIMESCALE = 10
N_EPISODES = 10000000
BATCH_SIZE = 1
HOST_MODE = HostMode.Value('DEDICATED')
# HOST = '35.247.27.224'
HOST = 'localhost'
LEARNING_RATE = 1e-4
eps = np.finfo(np.float32).eps.item()

pretrained_model = 'runs/Dec16_08-56-36_dockermaker/' + MODEL_FILENAME_FMT % START_EPISODE
model_blob = bucket.get_blob(pretrained_model)
pretrained_model = '/tmp/mdl.pt'
model_blob.download_to_filename(pretrained_model)

# Derivates.
DELAY_ENUM_TO_STEP = math.floor(TICKS_PER_OBSERVATION / N_DELAY_ENUMS)

if USE_CHECKPOINTS:
    writer = SummaryWriter()
    events_filename = writer.file_writer.event_writer._ev_writer._file_name
    log_dir = writer.file_writer.get_logdir()
    logger.info('Checkpointing to: {}'.format(log_dir))

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

        self.affine1d1 = nn.Linear(3, 32)
        self.affine1d2 = nn.Linear(32, 128)

        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1)

        self.affine2a = nn.Linear(128, 2)
        self.affine2b = nn.Linear(128, 2)
        self.affine2c = nn.Linear(128, 3)
        self.affine2d = nn.Linear(128, N_DELAY_ENUMS)
        self.affine2e = nn.Linear(128, 128)

    def forward(self, xa, xb, enemy_nonheroes, hidden):
        logger.debug('policy(xa={}, xb={}, enemy_nonheroes={})'.format(xa, xb, enemy_nonheroes))

        xa = self.affine1a(xa)
        xb = self.affine1b(xb)

        unit_embedding = []
        for unit_m in enemy_nonheroes:
            unit_m = torch.from_numpy(unit_m)
            xd1 = F.relu(self.affine1d1(unit_m))
            xd2 = self.affine1d2(xd1)
            unit_embedding.append(xd2)

        # Create the variable length embedding for use in LSTM and attention head.
        unit_embedding = torch.stack(unit_embedding)  # shape: (n_units, 128)

        # We max over unit dim to have a fixed output shape bc the LSTM needs to learn about these units.
        unit_max, _ = torch.max(unit_embedding, dim=0)  # shape: (128,)

        # Combine for LSTM.
        x = F.relu(xa + xb + unit_max)
        
        # LSTM
        x, hidden = self.rnn(x.unsqueeze(1), hidden)
        x = x.squeeze(1)

        # Heads.
        action_scores_x = self.affine2a(x)
        action_scores_y = self.affine2b(x)
        action_scores_enum = self.affine2c(x)
        action_delay_enum = self.affine2d(x)
        action_unit_attention = self.affine2e(x)  # shape: (1, 128)
        action_unit_attention = torch.mm(action_unit_attention, unit_embedding.t())  # shape (1, n)

        return {
                'x': F.softmax(action_scores_x, dim=1),
                'y': F.softmax(action_scores_y, dim=1),
                'enum': F.softmax(action_scores_enum, dim=1),
                'delay': F.softmax(action_delay_enum, dim=1),
                'unit': F.softmax(action_unit_attention, dim=1),
                }, hidden


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

if pretrained_model:
    policy.load_state_dict(torch.load(pretrained_model), strict=False)


def finish_episode(rewards, log_probs):
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


def get_hero_unit(state):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
            and unit.name == 'npc_dota_hero_nevermore':
            return unit
    raise ValueError("hero {} not found in state:\n{}".format(id, state))


class Actor:

    ENV_RETRY_DELAY = 5
    EXCEPTION_RETRIES = 5

    def __init__(self, config, host='127.0.0.1', port=13337, name=''):
        self.host = host
        self.port = port
        self.config = config
        self.name = name
        self.log_prefix = 'Actor {}: '.format(self.name)
        self.env = None
        self.channel = None

    def connect(self):
        if self.channel is None:  # TODO(tzaman) OR channel is closed? How?
            # Set up a channel.
            self.channel = Channel(self.host, self.port, loop=asyncio.get_event_loop())
            self.env = DotaServiceStub(self.channel)
            logger.info(self.log_prefix + 'Channel opened.')

    def disconnect(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.env = None
            logger.info(self.log_prefix + 'Channel closed.')

    async def __call__(self):
        # When an actor is being called it should first open up a channel. When a channel is opened
        # it makes sense to try to re-use it for this actor. So if a channel has already been
        # opened we should try to reuse.

        for i in range(self.EXCEPTION_RETRIES):
            try:
                while True:
                    self.connect()

                    # Wait for game to boot.
                    response = await asyncio.wait_for(self.env.reset(self.config), timeout=90)
                    initial_obs = response.world_state

                    if response.status == Status.Value('OK'):
                        break
                    else:
                        # Busy channel. Disconnect current and retry.
                        self.disconnect()
                        logger.info(self.log_prefix + "Service not ready, retrying in {}s.".format(
                            self.ENV_RETRY_DELAY))
                        await asyncio.sleep(self.ENV_RETRY_DELAY)

                return await self.call(obs=initial_obs)

            except Exception as e:
                logger.error(self.log_prefix + 'Exception call; retrying ({}/{}).:\n{}'.format(
                    i, self.EXCEPTION_RETRIES, e))
                traceback.print_exc()
                # We always disconnect the channel upon exceptions.
                self.disconnect()
            await asyncio.sleep(1)

    async def call(self, obs):
        logger.info(self.log_prefix + 'Starting game.')
        rewards = []
        log_probs = []
        hidden = None
        for step in range(N_STEPS):  # Steps/actions in the environment
            prev_obs = obs
            action, hidden = self.select_action(world_state=obs, hidden=hidden, step=step)
            logger.debug(self.log_prefix + 'action:{}'.format(action))

            log_probs.append({k: v['logprob'] for k, v in action.items() if 'logprob' in v})

            action_pb = self.action_to_pb(action=action, state=obs)

            response = await asyncio.wait_for(self.env.step(Action(action=action_pb)), timeout=15)
            if response.status != Status.Value('OK'):
                raise ValueError(self.log_prefix + 'Step reponse invalid:\n{}'.format(response))
            obs = response.world_state

            reward = get_reward(prev_obs=prev_obs, obs=obs)

            logger.debug(self.log_prefix + 'step={} reward={:.3f}\n'.format(step, sum(reward.values())))
            rewards.append(reward)

        await asyncio.wait_for(self.env.clear(Empty()), timeout=15)
        reward_sum = sum([sum(r.values()) for r in rewards])
        logger.info(self.log_prefix + 'Finished. reward_sum={:.2f}'.format(reward_sum))
        return rewards, log_probs


    def unit_matrix(self, state, hero_unit, unit_type=CMsgBotWorldState.UnitType.Value('LANE_CREEP')):
        # Prepopulate with invalid creep.
        handles = [-1]
        m = [np.array([0, 0, 0], dtype=np.float32)]

        for unit in state.units:
            if unit.team_id != hero_unit.team_id and unit.is_alive \
                and unit.unit_type == unit_type:
                rel_hp = (unit.health / unit.health_max)  # [0 (dead) : 1 (full hp)]
                distance_x = (hero_unit.location.x - unit.location.x) / 3000.
                distance_y = (hero_unit.location.y - unit.location.y) / 3000.
                m.append(np.array([rel_hp, distance_x, distance_y], dtype=np.float32))
                handles.append(unit.handle)
        return m, handles

    def select_action(self, world_state, hidden, step=None):
        actions = {}

        # Preprocess the state
        hero_unit = get_hero_unit(world_state)

        # Location Input
        location_state = np.array([hero_unit.location.x, hero_unit.location.y]) / 7000.  # maps the map between [-1 and 1]
        location_state = torch.from_numpy(location_state).float().unsqueeze(0)

        # Health and dotatime input
        hp_rel = 1. - (hero_unit.health / hero_unit.health_max) # Map between [0 and 1]
        dota_time_norm = dota_time = world_state.dota_time / 1200.  # Normalize by 20 minutes
        env_state = torch.from_numpy(np.array([hp_rel, dota_time_norm])).float().unsqueeze(0) 

        # Process the enemy_nonheroes
        enemy_nonheroes, handles = self.unit_matrix(state=world_state, hero_unit=hero_unit)

        probs, hidden = policy(
            xa=location_state,
            xb=env_state,
            enemy_nonheroes=enemy_nonheroes,
            hidden=hidden,
        )
            
        for k, prob in probs.items():
            m = Categorical(prob)
            action = m.sample()
            actions[k] = {'action': action.item(), 'prob': prob, 'logprob' :m.log_prob(action)}

        actions['unit']['handle'] = handles[actions['unit']['action']]

        return actions, hidden

    def action_to_pb(self, action, state):
        hero_unit = get_hero_unit(state)

        action_pb = CMsgBotWorldState.Action()
        action_pb.actionDelay = action['delay']['action'] * DELAY_ENUM_TO_STEP
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
            m.target = action['unit']['handle']
            m.once = False
            action_pb.attackTarget.CopyFrom(m)
        else:
            raise ValueError("unknown action {}".format(action_enum))
        return action_pb


def discount_rewards(rewards, gamma=0.99):
    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


async def main():
    config = Config(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
    )

    actors = [Actor(config=config, host=HOST, name=name) for name in range(BATCH_SIZE)]

    for episode in range(START_EPISODE, N_EPISODES):
        logger.info('=== Starting Episode {}.'.format(episode))
        all_rewards = []
        all_discounted_rewards = []
        all_log_probs = []

        start_time = time.time()

        actor_output = await asyncio.gather(*[a() for a in actors])

        # Loop over all distributed actors.
        for rewards, log_probs in actor_output:
            all_rewards.append(rewards)
            combined_rewards = [sum(r.values()) for r in rewards]
            discounted_rewards = discount_rewards(combined_rewards)
            all_discounted_rewards.extend(discounted_rewards)
            all_log_probs.extend(log_probs)

        time_per_batch = time.time() - start_time
        steps_per_s = len(all_log_probs) / time_per_batch

        loss = finish_episode(rewards=all_discounted_rewards, log_probs=all_log_probs)

        reward_counter = Counter()
        for b in all_rewards: # Jobs in a batch.
            for s in b: # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)
                
        reward_sum = sum(reward_counter.values())
        avg_reward = reward_sum / BATCH_SIZE
        logger.info('Episode={} avg_reward={:.2f} loss={:.3f}, steps/s={:.2f}'.format(
            episode, avg_reward, loss, steps_per_s))

        if USE_CHECKPOINTS:
            writer.add_scalar('steps per s', steps_per_s, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('mean_reward', avg_reward, episode)
            for k, v in reward_counter.items():
                writer.add_scalar('reward_{}'.format(k), v / BATCH_SIZE, episode)
            filename = MODEL_FILENAME_FMT % episode
            rel_path = os.path.join(log_dir, filename)
            torch.save(policy.state_dict(), rel_path)
            # Upload to GCP.
            blob = bucket.blob(rel_path)
            blob.upload_from_filename(filename=rel_path)  # Model
            blob = bucket.blob(events_filename)
            blob.upload_from_filename(filename=events_filename)  # Events file
        

if __name__ == '__main__':
    asyncio.run(main())
