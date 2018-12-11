import asyncio
from struct import unpack
async def worldstate_listener(port):
    while True:  # TODO(tzaman): finite retries.
        try:
            await asyncio.sleep(0.5)
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
        except ConnectionRefusedError:
            pass
        else:
            break
    while True:
        data = await reader.read(4)
        n_bytes = unpack("@I", data)[0]
        data = await reader.read(n_bytes) # Should we timeout for this?

loop = asyncio.get_event_loop()

tasks = asyncio.gather(
    worldstate_listener(12120),
    # worldstate_listener(12121),
)

loop.run_until_complete(tasks)