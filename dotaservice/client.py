import asyncio
import time

from grpclib.client import Channel

from protobuf.DotaService_grpc import DotaServiceStub
from protobuf.DotaService_pb2 import Action
from protobuf.DotaService_pb2 import Config
from protobuf.DotaAction_pb2 import DotaAction

async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 50051, loop=loop)
    stub = DotaServiceStub(channel)



    response = await stub.reset(Config())
    print('reset response:\ndotatime = ', response.world_state.dota_time)
    start_dotatime = response.world_state.dota_time

    start = time.time()
    nsteps = 1000
    for i in range(nsteps):
        dota_action = DotaAction(x=i, y=2, z=3)
        response = await stub.step(Action(action=dota_action))

        print('step response:\ndotatime = ', response.world_state.dota_time)
    end = time.time()
    dt = end-start
    print('dt= {} s'.format(end - start))
    print('{} steps/s'.format(float(nsteps) / dt))
    total_dotatime = response.world_state.dota_time - start_dotatime
    print('dota time passed:', total_dotatime)


if __name__ == '__main__':
    asyncio.run(main())
