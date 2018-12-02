from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from struct import unpack
import asyncio
import atexit
import glob
import json
import logging
import math
import os
import psutil
import re
import shutil
import signal
import subprocess
import time
import uuid

from aiohttp import web
from google.protobuf.json_format import MessageToDict
from grpclib.server import Server
import google.protobuf.text_format as txtf

from protobuf.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from protobuf.DotaService_grpc import DotaServiceBase
from protobuf.DotaService_pb2 import Observation

# logging.basicConfig(level=logging.DEBUG)  # This logging is a bit bananas
routes = web.RouteTableDef()

os.system("ps | grep dota2 | awk '{print $1}' | xargs kill -9")

CONSOLE_LOG_FILENAME = 'console.log'
TICKS_PER_OBSERVATION = 30
DOTA_PATH = '/Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game'
PORT_WORLDSTATE_RADIANT = 12120
PORT_WORLDSTATE_DIRE = 12121

# An enum from the game (should have been in the proto though).
DOTA_GAMERULES_STATE_PRE_GAME = 4
DOTA_GAMERULES_STATE_GAME_IN_PROGRESS = 5


if not os.path.exists(DOTA_PATH):
    raise ValueError('dota game path does not exist: {}'.format(DOTA_PATH))

BOTS_FOLDER_NAME = 'bots'
DOTA_BOT_PATH = os.path.join(DOTA_PATH, 'dota', 'scripts', 'vscripts', 'bots')

# Remove the dota bot directory
if os.path.exists(DOTA_BOT_PATH) or os.path.islink(DOTA_BOT_PATH):
    if os.path.isdir(DOTA_BOT_PATH) and not os.path.islink(DOTA_BOT_PATH):
        raise ValueError(
            'There is already a bots directory ({})! Please remove manually.'.format(DOTA_BOT_PATH))
    os.remove(DOTA_BOT_PATH)

GAME_ID = uuid.uuid1()
print('GAME_ID=', GAME_ID)

ACTION_FOLDER_ROOT = '/Volumes/ramdisk/'
if not os.path.exists(ACTION_FOLDER_ROOT):
    raise ValueError('Action folder does not exist. Please mount! ({})'.format(ACTION_FOLDER_ROOT))

SESSION_FOLDER = os.path.join(ACTION_FOLDER_ROOT, str(GAME_ID))
os.mkdir(SESSION_FOLDER)

BOT_PATH = os.path.join(SESSION_FOLDER, BOTS_FOLDER_NAME)
os.mkdir(BOT_PATH)

# Copy all the bot files into the action folder
for filename in glob.glob('../bot_script/*.lua'):
    shutil.copy(filename, BOT_PATH)

# Symlink DOTA to this folder
os.symlink(src=BOT_PATH, dst=DOTA_BOT_PATH)


def write_bot_data_file(filename_stem, data, atomic=False):
    filename = os.path.join(BOT_PATH, '{}.lua'.format(filename_stem))
    data = """
    -- THIS FILE IS AUTO GENERATED
    return '{data}'
    """.format(data=json.dumps(data, separators=(',',':')))
    if atomic:
        atomic_file_write(filename, data)
    else:
        with open(filename, 'w') as f:
            f.write(data)


config = {
    'game_id': str(GAME_ID),
    'ticks_per_observation': TICKS_PER_OBSERVATION,
}
write_bot_data_file(filename_stem='config_auto', data=config)


def atomic_file_write(filename, data):
    filename_tmp = "{}_".format(filename)
    f = open(filename_tmp, 'w')
    f.write(data)
    f.flush()
    os.fsync(f.fileno()) 
    f.close()
    os.rename(filename_tmp, filename)


class DotaService(DotaServiceBase):

    prev_time = None

    async def reset(self, stream):
        """reset method.

        This method should start up the dota game and the other required services.
        """
        print('DotaService::reset()')
        
        # Create all the processes here. 

        # TODO(tzaman)

        # We then have to wait for the first tick to come in
        # first_tick = await first_tick_future
        print('(py) DotaService::reset')#, first_tick=', first_tick)

        # We first wait for the lua config,
        print('(py) reset is awaiting lua config')
        lua_config = await lua_config_future
        print('(py) lua config received=', lua_config)

        # Cycle through the queue until its empty, then only using the latest worldstate.
        data = None
        try:
            while True:
                data = await asyncio.wait_for(worldstate_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            pass

        if data is None:
            raise ValueError('Worldstate queue empty while lua bot is ready!')

        self.prev_time = dotatime_to_ms(data.dota_time)

        # Now write the calibration file.
        config = {
            'calibration_ms': dotatime_to_ms(data.dota_time),
        }
        print('(py) writing live config=', config)
        write_bot_data_file(filename_stem='live_config_auto', data=config, atomic=True)

        # Return the reponse
        await stream.send_message(Observation(world_state=data))

    async def step(self, stream):
        print('DotaService::step()')

        request = await stream.recv_message()

        action = MessageToDict(request.action)

        # Add the dotatime to the dict for verification.
        action['dota_time'] = self.prev_time

        print('(python) action=', action)

        write_bot_data_file(filename_stem='action', data=action)

        # We've started to assume our queue will only have 1 item.
        data = await worldstate_queue.get()

        # Update the tick
        self.prev_time = dotatime_to_ms(data.dota_time)

        # Make sure indeed the queue is empty and we're entirely in sync.
        assert worldstate_queue.qsize() == 0
        
        # Return the reponse.
        await stream.send_message(Observation(world_state=data))


async def serve(server, *, host='127.0.0.1', port=50051):
    await server.start(host, port)
    print('Serving on {}:{}'.format(host, port))
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()


async def grpc_main(loop):
    server = Server([DotaService()], loop=loop)
    await serve(server)


def dotatime_to_ms(dotatime):
    return "{:09d}".format(math.floor(dotatime * 1000))


def kill_processes_and_children(pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


async def monitor_log():
    p = re.compile(r'LUARDY[ \t](\{.*\})')
    while True:
        filename = os.path.join(BOT_PATH, CONSOLE_LOG_FILENAME)
        if os.path.exists(filename):
            with open(filename) as f:
                for line in f:
                    m = p.search(line)
                    if m:
                        found = m.group(1)
                        lua_config = json.loads(found)
                        print('(py) lua_config = ', lua_config)
                        lua_config_future.set_result(lua_config)
                        return
        await asyncio.sleep(0.2)


async def record_replay(process):
    print("Starting to wait.")
    await asyncio.sleep(5)  # TODO(tzaman): just invoke after LUARDY signal.
    print('writing to stdin!')
    process.stdin.write(b"tv_record scripts/vscripts/bots/replay\n")
    await process.stdin.drain()


async def run_dota():
    script_path = os.path.join(DOTA_PATH, 'dota.sh')
    args = [
        script_path,
        "-botworldstatesocket_threaded",
        "-botworldstatetosocket_dire 12121",
        "-botworldstatetosocket_frames {}".format(TICKS_PER_OBSERVATION),
        "-botworldstatetosocket_radiant 12120",
        "-con_logfile scripts/vscripts/bots/{}".format(CONSOLE_LOG_FILENAME),
        "-con_timestamp",
        "-console",
        "-dedicated",
        "-fill_with_bots",
        "-insecure",
        "-noip",
        "-nowatchdog",  # WatchDog will quit the game if e.g. the lua api takes a few seconds.
        "+clientport 27006",  # Relates to steam client.
        "+dota_1v1_skip_strategy 1",  # doesn't work icm `-fill_with_bots`
        "+dota_auto_surrender_all_disconnected_timeout 0",  # Used when `dota_surrender_on_disconnect` is 1
        "+dota_bot_practice_gamemode 11",  # Mid Only -> doesn't work icm `-fill_with_bots`
        "+dota_force_gamemode 11",  # Mid Only -> doesn't work icm `-fill_with_bots`
        "+dota_start_ai_game 1",
        "+dota_surrender_on_disconnect 0",
        "+host_force_frametime_to_equal_tick_interval 1",
        "+host_timescale 10",
        "+host_writeconfig 1",
        "+hostname dotaservice",
        "+map start",  # the `start` map works with replays when dedicated, map `dota` doesn't.
        "+sv_cheats 1",
        "+sv_hibernate_when_empty 0",
        "+sv_lan 1",
        "+tv_delay 0 ",
        "+tv_enable 1",
        "+tv_title {}".format(GAME_ID),
    ]
    create = asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        # stdout=asyncio.subprocess.PIPE,
        # stderr=asyncio.subprocess.PIPE,
    )
    process = await create


    task_record_replay = asyncio.create_task(begin(process=process))

    task_monitor_log = asyncio.create_task(monitor_log())

    try:
        await process.wait()
    except asyncio.CancelledError:
        kill_processes_and_children(pid=process.pid)
        raise


async def data_from_reader(reader):
    # Receive the package length.
    data = await reader.read(4)
    n_bytes = unpack("@I", data)[0]
    # Receive the payload given the length.
    data = await asyncio.wait_for(reader.read(n_bytes), timeout=5.0)
    # Decode the payload.
    parsed_data = CMsgBotWorldState()
    parsed_data.ParseFromString(data)
    dotatime = parsed_data.dota_time
    gamestate = parsed_data.game_state
    ms = dotatime_to_ms(dotatime)
    print('(py) worldstate @dotatime={} @ms={} @gamestate={}'.format(dotatime, ms, gamestate))
    return parsed_data


async def worldstate_listener(port):
    await asyncio.sleep(2)
    reader, writer = await asyncio.open_connection('127.0.0.1', port)#, loop=loop)
    try:
        while True:
            # This reader is always going to need to keep going to keep the buffers clean.
            parsed_data = await data_from_reader(reader)
            worldstate_queue.put_nowait(parsed_data)
    except asyncio.CancelledError:
        raise


loop = asyncio.get_event_loop()

worldstate_queue = asyncio.Queue(loop=loop)

worldstate_calibration_tick_future = loop.create_future()

lua_config_future = loop.create_future()

tasks =  asyncio.gather(
    run_dota(),
    grpc_main(loop),
    worldstate_listener(port=PORT_WORLDSTATE_RADIANT),
    # worldstate_listener(port=PORT_WORLDSTATE_DIRE),
)



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
