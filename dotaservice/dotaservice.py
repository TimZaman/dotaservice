from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from struct import unpack
import asyncio
import atexit
import math
import os
import psutil
import shutil
import signal
import subprocess
import time
import json

import google.protobuf.text_format as txtf
from grpclib.server import Server
from google.protobuf.json_format import MessageToDict

from protobuf.DotaService_grpc import DotaServiceBase
from protobuf.DotaService_pb2 import Observation
from protobuf.CMsgBotWorldState_pb2 import CMsgBotWorldState

# global_test_var = ''

TICKS_PER_SECOND = 30
DOTA_PATH = '/Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game'
PORT_WORLDSTATE_RADIANT = 12120
PORT_WORLDSTATE_DIRE = 12121

if not os.path.exists(DOTA_PATH):
    raise ValueError('dota game path does not exist: {}'.format(DOTA_PATH))

BOT_PATH = os.path.join(DOTA_PATH, 'dota', 'scripts', 'vscripts', 'bots')

if os.path.exists(BOT_PATH):
    shutil.rmtree(BOT_PATH)
shutil.copytree('../bot_script', BOT_PATH)

reader = None

ACTION_FOLDER = '/Volumes/ramdisk/'
GAME_ID = 'my_game_id'
if not os.path.exists(DOTA_PATH):
    raise ValueError('Action folder does not exist. Please mount! ({})'.format(ACTION_FOLDER))
GAME_ACTION_FOLDER = os.path.join(ACTION_FOLDER, GAME_ID)
if not os.path.exists(DOTA_PATH):
    os.mkdir(GAME_ACTION_FOLDER)


class DotaService(DotaServiceBase):

    async def Step(self, stream):
        print('DotaService::Step()')
        request = await stream.recv_message()
        print('  request={}'.format(request))
        # empty_worldstate = CMsgBotWorldState()

        # d = {'foo': 1337}
        tick = 0
        filename = os.path.join(GAME_ACTION_FOLDER, "{}.lua".format(tick))
        data_dict = MessageToDict(request)
        # print('type(data_js):', type(data_js))
        # print('data_js:', data_js)
        # data_js = 
        data = "data = '{}'".format(json.dumps(data_dict, separators=(',',':')))
        print('data=', data)
        with open(filename, 'w') as f:
            f.write(data)

        # Somehow i need the world state from the bot here

        # data = await data_from_reader(reader=reader)
        data = CMsgBotWorldState()  # Dummy
        # print('data=', data)
        await stream.send_message(Observation(world_state=data))


async def serve(server, *, host='127.0.0.1', port=50051):
    await server.start(host, port)
    print('Serving on {}:{}'.format(host, port))
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()


async def grpc_main():
    server = Server([DotaService()], loop=asyncio.get_event_loop())
    await serve(server)





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
        # "-botworldstatesocket_threaded",
        "-botworldstatetosocket_dire 12121",
        "-botworldstatetosocket_frames 30",
        "-botworldstatetosocket_radiant 12120",
        # "-console,",
        "-dedicated",
        # "-dev",  # Not sure what this does
        "-fill_with_bots",
        "-host_force_frametime_to_equal_tick_interval 1",
        "-insecure",
        "-nowatchdog",  # WatchDog will quit the game if e.g. the lua api takes a few seconds.
        "+clientport 27006",  # Relates to steam client.
        "+dota_1v1_skip_strategy 1",  # doesn't work icm `-fill_with_bots`
        "+dota_auto_surrender_all_disconnected_timeout 0",
        "+dota_bot_practice_gamemode 11",  # Mid Only -> doesn't work icm `-fill_with_bots`
        "+dota_force_gamemode 11",  # Mid Only -> doesn't work icm `-fill_with_bots`
        "+dota_start_ai_game 1",
        "+dota_surrender_on_disconnect 0",
        "+host_timescale 10",
        "+map start",  # the `start` map works with replays when dedicated, map `dota` doesn't.
        "+sv_cheats 1",
        "+sv_hibernate_when_empty 0",
        "+sv_lan 1",
        "+tv_autorecord 1",
        "+tv_delay 0 ",
        "+tv_dota_auto_record_stressbots 1",
        "+tv_enable 1",
    ]
    create = asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        # stdout=asyncio.subprocess.PIPE,
        # stderr=asyncio.subprocess.PIPE,
    )
    proc = await create
    try:
        await proc.wait()
    except asyncio.CancelledError:
        kill_processes_and_children(pid=proc.pid)
        raise


async def data_from_reader(reader):
    port = None  # HACK
    print('data_from_reader')
    # Receive the package length.
    data = await asyncio.wait_for(reader.read(4), timeout=5.0)
    # eternity(), timeout=1.0)
    n_bytes = unpack("@I", data)[0]
    # Receive the payload given the length.
    data = await asyncio.wait_for(reader.read(n_bytes), timeout=5.0)
    # Decode the payload.
    parsed_data = CMsgBotWorldState()
    parsed_data.ParseFromString(data)
    dotatime = parsed_data.dota_time
    tick = dotatime_to_tick(dotatime)
    # global_test_var = tick
    print('@port{} tick: {}'.format(port, tick))
    return parsed_data

async def worldstate_listener(port):
    # global global_test_var
    # global reader
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
            global_test_var = tick
            print('@port{} tick: {}'.format(port, tick))
    except asyncio.CancelledError:
        raise


tasks =  asyncio.gather(
    run_dota(),
    grpc_main(),
    worldstate_listener(port=PORT_WORLDSTATE_RADIANT),
    worldstate_listener(port=PORT_WORLDSTATE_DIRE),
)
# exit()

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
