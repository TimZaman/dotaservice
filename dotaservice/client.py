import asyncio

from grpclib.client import Channel

from protobuf.DotaService_grpc import DotaServiceStub
from protobuf.DotaService_pb2 import Action
from protobuf.DotaAction_pb2 import DotaAction

async def main():
    loop = asyncio.get_event_loop()
    channel = Channel('127.0.0.1', 50051, loop=loop)
    stub = DotaServiceStub(channel)

    dota_action = DotaAction(x=1, y=2, z=3)
    response = await stub.Step(Action(action=dota_action))

    print('Response:\ndotatime = ', response.world_state.dota_time)
    # print('--- Reponse ---\n{}'.format(response))


if __name__ == '__main__':
    asyncio.run(main())
