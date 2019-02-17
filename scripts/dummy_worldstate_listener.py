import asyncio
from struct import unpack

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

async def start(host='127.0.0.1', port=30001):
    reader, writer = await asyncio.open_connection(host, port)
    print('reader=', reader)
    print('writer=', writer)
    while True:
        x = await reader.read()
        print(x)
        continue
        data = await reader.read(4)
        if data == b'':
            continue
        print('data=', data)
        n_bytes = unpack("@I", data)[0]
        print('n_bytes=', n_bytes)
        # Receive the payload given the length.
        data = await asyncio.wait_for(reader.read(n_bytes), timeout=2)
        print(data)
        world_state = CMsgBotWorldState()
        world_state.ParseFromString(data)
        print(world_state.dota_time)

asyncio.run(start())