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
from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 13337, loop=loop)
    env = DotaServiceStub(channel)

    for host_timescale in [1, 5, 10]:
        for ticks_per_observation in [1, 5, 10, 30]:
            config = Config(
                host_timescale=host_timescale,
                ticks_per_observation=ticks_per_observation,
                render=False)

            # print('(client) episode: %s' % e)
            start = time()
            observation = await env.reset(config)
            start_dt = time() - start
            # print('start dt=', time()-start)
            # print('reset observation:\ndotatime = ', observation.world_state.dota_time)
            start_dotatime = observation.world_state.dota_time
            dts = 0.
            nsteps = 10000
            for i in range(nsteps):
                # print('(client) step %s' % i)
                action = CMsgBotWorldState.Action()
                action.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_TO_POSITION')
                m = CMsgBotWorldState.Action.MoveToLocation()
                m.location.x = math.sin(observation.world_state.dota_time) * 500 -1000
                m.location.y = math.cos(observation.world_state.dota_time) * 500 -1000
                m.location.z = 0
                action.moveToLocation.CopyFrom(m)
                # print('action=', action)
                start = time()
                observation = await env.step(Action(action=action))
                dts += (time()-start)
                # print('dt=', time()-start)
                # print('(client) sent action for dotatime=', observation.world_state.dota_time)
            print('| {} | {} | {} | {} |'.format(host_timescale, ticks_per_observation, int(start_dt*1000), int(1000.*dts/nsteps)))


if __name__ == '__main__':
    asyncio.run(main())
