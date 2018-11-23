
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from struct import unpack
import asyncio
import atexit
import math
import os
import psutil
import signal
import subprocess
import time

import protobuf.CMsgBotWorldState_pb2 as _pb
import google.protobuf.text_format as txtf

TICKS_PER_SECOND = 30
DOTA_PATH = '/Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game'
PORT_WORLDSTATE_RADIANT = 12120
PORT_WORLDSTATE_DIRE = 12121


def dotatime_to_tick(dotatime):
    return math.floor(dotatime * TICKS_PER_SECOND)


def kill_processes_and_children(pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


async def run_dota():
    script_path = os.path.join(DOTA_PATH, 'dota.sh')
    args = [
        script_path,
        "-botworldstatesocket_threaded",
        "-botworldstatetosocket_dire 12121",
        "-botworldstatetosocket_frames 5",
        "-botworldstatetosocket_radiant 12120",
        "-console,",
        "-dedicated",
        "-fill_with_bots",
        "-host_force_frametime_to_equal_tick_interval 1",
        "-insecure",
        "+clientport 27006",
        "+dota_auto_surrender_all_disconnected_timeout 180",
        "+host_timescale 1",
        "+map dota",
        "+sv_cheats 1",
        "+sv_hibernate_when_empty 0",
        "+sv_lan 1",
    ]
    create = asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    proc = await create
    try:
        await proc.wait()
    except asyncio.CancelledError:
        kill_processes_and_children(pid=proc.pid)
        raise


async def worldstate_listener(port):
    print('creating worldstate_listener @ port %s' % port)
    await asyncio.sleep(5)
    reader, writer = await asyncio.open_connection('127.0.0.1', port)#, loop=loop)
    try:
        while True:
            # Receive the package length.
            data = await asyncio.wait_for(reader.read(4), timeout=5.0)
            # eternity(), timeout=1.0)
            n_bytes = unpack("@I", data)[0]
            # Receive the payload given the length.
            data = await asyncio.wait_for(reader.read(n_bytes), timeout=5.0)
            # Decode the payload.
            parsed_data = _pb.CMsgBotWorldState()
            parsed_data.ParseFromString(data)
            dotatime = parsed_data.dota_time
            tick = dotatime_to_tick(dotatime)
            print('@port{} tick: {}'.format(port, tick))
    except asyncio.CancelledError:
        raise


tasks =  asyncio.gather(
    run_dota(),
    worldstate_listener(port=PORT_WORLDSTATE_RADIANT),
    worldstate_listener(port=PORT_WORLDSTATE_DIRE),
)

loop = asyncio.get_event_loop()

try:
    loop.run_until_complete(tasks)
except KeyboardInterrupt:
    # Optionally show a message if the shutdown may take a while
    print("Attempting graceful shutdown, press Ctrl+C again to exit…", flush=True)

    # Do not show `asyncio.CancelledError` exceptions during shutdown
    # (a lot of these may be generated, skip this if you prefer to see them)
    def shutdown_exception_handler(loop, context):
        if "exception" not in context \
        or not isinstance(context["exception"], asyncio.CancelledError):
            loop.default_exception_handler(context)
    loop.set_exception_handler(shutdown_exception_handler)

    # Handle shutdown gracefully by waiting for all tasks to be cancelled
    tasks = asyncio.gather(*asyncio.Task.all_tasks(loop=loop), loop=loop, return_exceptions=True)
    tasks.add_done_callback(lambda t: loop.stop())
    tasks.cancel()

    # Keep the event loop running until it is either destroyed or all
    # tasks have really terminated
    while not tasks.done() and not loop.is_closed():
        loop.run_forever()
finally:
    loop.close()
