import asyncio
import time
import math

from grpclib.client import Channel
from google.protobuf.json_format import MessageToDict

from protobuf.DotaService_grpc import DotaServiceStub
from protobuf.DotaService_pb2 import Action
from protobuf.DotaService_pb2 import Config

from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 13337, loop=loop)
    env = DotaServiceStub(channel)


    for e in range(2):
        print('(client) episode: %s' % e)
        observation = await env.reset(Config())
        print('reset observation:\ndotatime = ', observation.world_state.dota_time)
        start_dotatime = observation.world_state.dota_time
        for i in range(100):
            print('(client) step %s' % i)
            action = CMsgBotWorldState.Action()
            action.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_TO_POSITION')
            m = CMsgBotWorldState.Action.MoveToLocation()
            m.location.x = math.sin(observation.world_state.dota_time) * 500 -1000
            m.location.y = math.cos(observation.world_state.dota_time) * 500 -1000
            m.location.z = 0
            action.moveToLocation.CopyFrom(m)
            print('action=', action)
            observation = await env.step(Action(action=action))
            print('(client) sent action for dotatime=', observation.world_state.dota_time)


if __name__ == '__main__':
    asyncio.run(main())
