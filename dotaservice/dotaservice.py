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
import pkg_resources
import re
import shutil
import signal
import subprocess
import time
import uuid

from google.protobuf.json_format import MessageToDict
from grpclib.server import Server
import psutil

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_grpc import DotaServiceBase
from dotaservice.protos.DotaService_pb2 import Observation

# logging.basicConfig(level=logging.DEBUG)  # This logging is a bit bananas

# An enum from the game (should have been in the proto [dota_gcmessages_common.proto?] though).
DOTA_GAMERULES_STATE_PRE_GAME = 4
DOTA_GAMERULES_STATE_GAME_IN_PROGRESS = 5

# Static environment variables.
DOTA_PATH = '/Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game'
BOTS_FOLDER_NAME = 'bots'
DOTA_BOT_PATH = os.path.join(DOTA_PATH, 'dota', 'scripts', 'vscripts', BOTS_FOLDER_NAME)
ACTION_FOLDER_ROOT = '/Volumes/ramdisk/'
CONSOLE_LOG_FILENAME = 'console.log'
PORT_WORLDSTATE_RADIANT = 12120
PORT_WORLDSTATE_DIRE = 12121
LUA_FILES_GLOB = pkg_resources.resource_filename('dotaservice', 'lua/*.lua')

# Initial environment assertions.
if not os.path.exists(DOTA_PATH):
    raise ValueError('dota game path does not exist: {}'.format(DOTA_PATH))

if not os.path.exists(ACTION_FOLDER_ROOT):
    raise ValueError('Action folder does not exist. Please mount! ({})'.format(ACTION_FOLDER_ROOT))


def kill_processes_and_children(pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


class DotaGame(object):

    CONFIG_FILENAME = 'config_auto'

    def __init__(self, host_timescale, ticks_per_observation, render, game_id=None):
        self._dota_time = None

        self.host_timescale = host_timescale
        self.ticks_per_observation = ticks_per_observation
        self.render = render

        self.game_id = game_id
        if not self.game_id:
            self.game_id = str(uuid.uuid1())

        self.bot_path = self._create_bot_path(game_id=self.game_id)

        self.worldstate_queue = asyncio.Queue(loop=asyncio.get_event_loop())
        self.lua_config_future = asyncio.get_event_loop().create_future()

        self._write_config()

    def _write_config(self):
        # Write out the game configuration.
        config = {
            'game_id': self.game_id,
            'ticks_per_observation': self.ticks_per_observation,
        }
        self.write_bot_data_file(filename_stem=self.CONFIG_FILENAME, data=config)

    @property
    def dota_time(self):
        return self._dota_time

    @dota_time.setter
    def dota_time(self, value):
        # TODO(tzaman): check that new value is larger than old one.
        if self._dota_time is not None and value < self._dota_time:
            raise ValueError('New dota time {} is larger than the old one {}'.format(
                value, self._dota_time))
        self._dota_time = value

    def write_bot_data_file(self, filename_stem, data):
        """Write a file to lua to that the bot can read it.

        Although writing atomicly would prevent bad reads, we just catch the bad reads in the
        dota bot client.
        """
        filename = os.path.join(self.bot_path, '{}.lua'.format(filename_stem))
        data = """
        -- THIS FILE IS AUTO GENERATED
        return '{data}'
        """.format(data=json.dumps(data, separators=(',', ':')))
        with open(filename, 'w') as f:
            f.write(data)

    @staticmethod
    def _create_bot_path(game_id):
        """Remove DOTA's bots subdirectory or symlink and update it with our own."""
        print('(py) create_bot_path(game_id=%s', game_id)
        if os.path.exists(DOTA_BOT_PATH) or os.path.islink(DOTA_BOT_PATH):
            if os.path.isdir(DOTA_BOT_PATH) and not os.path.islink(DOTA_BOT_PATH):
                raise ValueError('There is already a bots directory ({})! Please remove manually.'.
                                 format(DOTA_BOT_PATH))
            os.remove(DOTA_BOT_PATH)
        SESSION_FOLDER = os.path.join(ACTION_FOLDER_ROOT, str(game_id))
        os.mkdir(SESSION_FOLDER)
        bot_path = os.path.join(SESSION_FOLDER, BOTS_FOLDER_NAME)
        os.mkdir(bot_path)

        # Copy all the bot files into the action folder.
        lua_files = glob.glob(LUA_FILES_GLOB)
        assert len(lua_files) == 5
        for filename in lua_files:
            shutil.copy(filename, bot_path)

        # Finally, symlink DOTA to this folder.
        os.symlink(src=bot_path, dst=DOTA_BOT_PATH)
        return bot_path

    async def monitor_log(self):
        print('@monitor_log')
        p = re.compile(r'LUARDY[ \t](\{.*\})')
        while True:  # TODO(tzaman): probably just retry 10x sleep(0.5) then bust?
            filename = os.path.join(self.bot_path, CONSOLE_LOG_FILENAME)
            if os.path.exists(filename):
                with open(filename) as f:
                    for line in f:
                        m = p.search(line)
                        if m:
                            found = m.group(1)
                            lua_config = json.loads(found)
                            print('(py) lua_config = ', lua_config)
                            self.lua_config_future.set_result(lua_config)
                            return
            await asyncio.sleep(0.2)

    async def record_replay(self, process):
        """Starts the stdin command.
        """
        print('@record_replay')
        await asyncio.sleep(5)  # TODO(tzaman): just invoke after LUARDY signal?
        process.stdin.write(b"tv_record scripts/vscripts/bots/replay\n")
        if self.render:
            # If we want to render, let the 'render player' join the spectator team.
            # TODO(tzaman): does this work at all?
            await asyncio.sleep(20)  # TODO(tzaman): come up with something smart.
            process.stdin.write(b"jointeam spec\n")
        await process.stdin.drain()

    async def run(self):
        # Start the worldstate listener(s).
        asyncio.create_task(self._run_dota())
        asyncio.create_task(self._worldstate_listener(port=PORT_WORLDSTATE_RADIANT))
        # asyncio.create_task(self.worldstate_listener(port=PORT_WORLDSTATE_DIRE))

    async def _run_dota(self):
        script_path = os.path.join(DOTA_PATH, 'dota.sh')
        args = [
            script_path,
            "-botworldstatesocket_threaded",
            "-botworldstatetosocket_dire {}".format(PORT_WORLDSTATE_RADIANT),
            "-botworldstatetosocket_frames {}".format(self.ticks_per_observation),
            "-botworldstatetosocket_radiant {}".format(PORT_WORLDSTATE_RADIANT),
            "-con_logfile scripts/vscripts/bots/{}".format(CONSOLE_LOG_FILENAME),
            "-con_timestamp",
            "-console",
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
            "+host_timescale {}".format(self.host_timescale),
            "+host_writeconfig 1",
            "+hostname dotaservice",
            "+map start",  # the `start` map works with replays when dedicated, map `dota` doesn't.
            "+sv_cheats 1",
            "+sv_hibernate_when_empty 0",
            "+sv_lan 1",
            "+tv_delay 0 ",
            "+tv_enable 1",
            "+tv_title {}".format(self.game_id),
        ]
        if not self.render:
            args.append('-dedicated')
        create = asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            # stdout=asyncio.subprocess.PIPE,
            # stderr=asyncio.subprocess.PIPE,
        )
        process = await create

        task_record_replay = asyncio.create_task(self.record_replay(process=process))
        task_monitor_log = asyncio.create_task(self.monitor_log())

        await process.wait()

        try:
            await process.wait()
        except asyncio.CancelledError:
            kill_processes_and_children(pid=process.pid)
            raise

    @staticmethod
    async def _data_from_reader(reader):
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
        print('(py) worldstate @ dotatime={}, gamestate={}'.format(dotatime, gamestate))
        return parsed_data

    async def _worldstate_listener(self, port):
        while True:  # TODO(tzaman): finite retries.
            try:
                await asyncio.sleep(0.5)
                reader, writer = await asyncio.open_connection('127.0.0.1', port)
            except ConnectionRefusedError:
                pass
            else:
                break
        try:
            while True:
                # This reader is always going to need to keep going to keep the buffers clean.
                parsed_data = await self._data_from_reader(reader)
                self.worldstate_queue.put_nowait(parsed_data)
        except asyncio.CancelledError:
            raise


class DotaService(DotaServiceBase):

    dota_game = None

    async def reset(self, stream):
        """reset method.

        This method should start up the dota game and the other required services.
        """
        print('DotaService::reset()')

        config = await stream.recv_message()
        print('config=\n', config)

        # Kill any previously running dota processes. # TODO(tzaman): do this cleanly.
        os.system("ps | grep dota2 | awk '{print $1}' | xargs kill -9")

        # Create a new dota game instance.
        # TODO(tzaman): kill previous dota game? or implicit through __del__?
        self.dota_game = DotaGame(
            host_timescale=config.host_timescale,
            ticks_per_observation=config.ticks_per_observation,
            render=config.render,
            game_id=config.game_id,
        )

        # Start dota.
        asyncio.create_task(self.dota_game.run())

        # We first wait for the lua config. TODO(tzaman): do this in DotaGame?
        print('(py) reset is awaiting lua config')
        lua_config = await self.dota_game.lua_config_future
        print('(py) lua config received=', lua_config)

        # Cycle through the queue until its empty, then only using the latest worldstate.
        data = None
        try:
            while True:
                data = await asyncio.wait_for(self.dota_game.worldstate_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            pass

        if data is None:
            raise ValueError('Worldstate queue empty while lua bot is ready!')

        self.dota_game.dota_time = data.dota_time

        # Now write the calibration file.
        config = {
            'calibration_dota_time': data.dota_time,
        }
        print('(py) writing live config=', config)
        self.dota_game.write_bot_data_file(filename_stem='live_config_auto', data=config)

        # Return the reponse
        await stream.send_message(Observation(world_state=data))

    async def step(self, stream):
        print('DotaService::step()')

        request = await stream.recv_message()

        action = MessageToDict(request.action)

        # Add the dotatime to the dict for verification.
        action['dota_time'] = self.dota_game.dota_time

        print('(python) action=', action)

        self.dota_game.write_bot_data_file(filename_stem='action', data=action)

        # We've started to assume our queue will only have 1 item.
        data = await self.dota_game.worldstate_queue.get()

        # Update the tick
        self.dota_game.dota_time = data.dota_time

        # Make sure indeed the queue is empty and we're entirely in sync.
        assert self.dota_game.worldstate_queue.qsize() == 0

        # Return the reponse.
        await stream.send_message(Observation(world_state=data))


async def serve(server, *, host='127.0.0.1', port=13337):
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


def main():
    loop = asyncio.get_event_loop()
    tasks = grpc_main(loop)

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
        tasks = asyncio.gather(
            *asyncio.Task.all_tasks(loop=loop), loop=loop, return_exceptions=True)
        tasks.add_done_callback(lambda t: loop.stop())
        tasks.cancel()

        # Keep the event loop running until it is either destroyed or all
        # tasks have really terminated
        while not tasks.done() and not loop.is_closed():
            loop.run_forever()
    finally:
        loop.close()
