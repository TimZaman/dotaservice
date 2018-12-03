import asyncio
from time import time
import math
import uuid

from grpclib.client import Channel
from google.protobuf.json_format import MessageToDict

from protobuf.DotaService_grpc import DotaServiceStub
from protobuf.DotaService_pb2 import Action
from protobuf.DotaService_pb2 import Config

from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

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


async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 13337, loop=loop)
    env = DotaServiceStub(channel)

    config = Config(
        host_timescale=10,
        ticks_per_observation=30,
        render=False,
    )

    nsteps = 10000
    nepisodes = 10

    for e in range(nepisodes):
        observation = await env.reset(config)
        rewards = get_rewards(observation)

        for i in range(nsteps):
            action = CMsgBotWorldState.Action()
            action.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
            m = CMsgBotWorldState.Action.MoveToLocation()
            m.location.x = math.sin(observation.world_state.dota_time) * 500 - 1000
            m.location.y = math.cos(observation.world_state.dota_time) * 500 - 1000
            m.location.z = 0
            action.moveToLocation.CopyFrom(m)
            observation = await env.step(Action(action=action))
            rewards = get_rewards(observation)

            print('t={:.2f}, rewards: {}'.format(observation.world_state.dota_time, rewards))


if __name__ == '__main__':
    asyncio.run(main())
