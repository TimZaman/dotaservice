import asyncio
import time

from grpclib.client import Channel
from google.protobuf.json_format import MessageToDict

from protobuf.DotaService_grpc import DotaServiceStub
from protobuf.DotaService_pb2 import Action
from protobuf.DotaService_pb2 import Config
# from protobuf.DotaAction_pb2 import DotaAction

from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 13337, loop=loop)
    env = DotaServiceStub(channel)

    response = await env.reset(Config())
    print('reset response:\ndotatime = ', response.world_state.dota_time)
    start_dotatime = response.world_state.dota_time

    start = time.time()
    nsteps = 100000
    # for i in range(nsteps):
    while True:
        action = CMsgBotWorldState.Action()
        action.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_TO_POSITION')
        m = CMsgBotWorldState.Action.MoveToLocation()
        m.location.x = 5
        m.location.y = 6
        m.location.z = 7
        action.moveToLocation.CopyFrom(m)

        print('action=', action)
        observation = await env.step(Action(action=action))

        print('(client) sent action for dotatime=', observation.world_state.dota_time)
    end = time.time()
    dt = end-start
    print('dt= {} s'.format(end - start))
    print('{} steps/s'.format(float(nsteps) / dt))
    total_dotatime = response.world_state.dota_time - start_dotatime
    print('dota time passed:', total_dotatime)


if __name__ == '__main__':
    asyncio.run(main())
