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

# data = CMsgBotWorldState()
# print('data=', data)
# print(MessageToDict(data))

# # from protobuf.dota_gcmessages_common_bot_script_pb2 import Action

# data = CMsgBotWorldState.Action()

# data.actionType = 2

# print('data=', data)
# print(MessageToDict(data))

# exit()

async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 50051, loop=loop)
    env = DotaServiceStub(channel)

    response = await env.reset(Config())
    print('reset response:\ndotatime = ', response.world_state.dota_time)
    start_dotatime = response.world_state.dota_time

    start = time.time()
    nsteps = 1000
    for i in range(nsteps):
        action = CMsgBotWorldState.Action()#x=i, y=2, z=3)
        action.actionType = 2
        observation = await env.step(Action(action=action))

        print('step observation:\ndotatime = ', observation.world_state.dota_time)
    end = time.time()
    dt = end-start
    print('dt= {} s'.format(end - start))
    print('{} steps/s'.format(float(nsteps) / dt))
    total_dotatime = response.world_state.dota_time - start_dotatime
    print('dota time passed:', total_dotatime)


if __name__ == '__main__':
    asyncio.run(main())
