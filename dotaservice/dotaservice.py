from struct import unpack
from sys import platform
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

from google.protobuf.message import DecodeError
from google.protobuf.json_format import MessageToDict
from grpclib.server import Server

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_grpc import DotaServiceBase
from dotaservice.protos.DotaService_pb2 import Observation

LUA_FILES_GLOB = pkg_resources.resource_filename('dotaservice', 'lua/*.lua')

# logging.basicConfig(level=logging.DEBUG)  # This logging is a bit overwhelming


def kill_processes_and_children(pid, sig=signal.SIGTERM):
    # TODO(tzaman): removed problematic `psutil`. Just use `os.$`?
    pass


def verify_game_path(game_path):
    if not os.path.exists(game_path):
        raise ValueError("Game path '{}' does not exist.".format(game_path))
    if not os.path.isdir(game_path):
        raise ValueError("Game path '{}' is not a directory.".format(game_path))
    dota_script = os.path.join(game_path, DotaGame.DOTA_SCRIPT_FILENAME)
    if not os.path.isfile(dota_script):
        raise ValueError("Dota executable '{}' is not a file.".format(dota_script))
    if not os.access(dota_script, os.X_OK):
        raise ValueError("Dota executable '{}' is not executable.".format(dota_script))


class DotaGame(object):

    ACTION_FILENAME = 'action'
    BOTS_FOLDER_NAME = 'bots'
    CONFIG_FILENAME = 'config_auto'
    CONSOLE_LOG_FILENAME = 'console.log'
    DOTA_SCRIPT_FILENAME = 'dota.sh'
    LIVE_CONFIG_FILENAME = 'live_config_auto'
    PORT_WORLDSTATE_DIRE = 12121
    PORT_WORLDSTATE_RADIANT = 12120
    RE_DEMO =  re.compile(r'playdemo[ \t](.*dem)')
    RE_LUARDY = re.compile(r'LUARDY[ \t](\{.*\})')

    def __init__(self,
                 dota_path,
                 action_folder,
                 host_timescale,
                 ticks_per_observation,
                 render,
                 game_id=None):
        self.dota_path = dota_path
        self.action_folder = action_folder
        self.host_timescale = host_timescale
        self.ticks_per_observation = ticks_per_observation
        self.render = render
        self.game_id = game_id
        if not self.game_id:
            self.game_id = str(uuid.uuid1())
        self._dota_time = None
        self.dota_bot_path = os.path.join(self.dota_path, 'dota', 'scripts', 'vscripts',
                                          self.BOTS_FOLDER_NAME)
        self.bot_path = self._create_bot_path()
        self.worldstate_queue = asyncio.Queue(loop=asyncio.get_event_loop())
        self.lua_config_future = asyncio.get_event_loop().create_future()
        self._write_config()
        self.process = None
        self.demo_path_rel = None

    def _write_config(self):
        # Write out the game configuration.
        config = {
            'game_id': self.game_id,
            'ticks_per_observation': self.ticks_per_observation,
        }
        self.write_static_config(data=config)

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

    def write_static_config(self, data):
        self._write_bot_data_file(filename_stem=self.CONFIG_FILENAME, data=data)

    def write_live_config(self, data):
        self._write_bot_data_file(filename_stem=self.LIVE_CONFIG_FILENAME, data=data)
        
    def write_action(self, data):
        self._write_bot_data_file(filename_stem=self.ACTION_FILENAME, data=data)

    def _write_bot_data_file(self, filename_stem, data):
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

    def _create_bot_path(self):
        """Remove DOTA's bots subdirectory or symlink and update it with our own."""
        if os.path.exists(self.dota_bot_path) or os.path.islink(self.dota_bot_path):
            if os.path.isdir(self.dota_bot_path) and not os.path.islink(self.dota_bot_path):
                raise ValueError(
                    'There is already a bots directory ({})! Please remove manually.'.format(
                        self.dota_bot_path))
            os.remove(self.dota_bot_path)
        session_folder = os.path.join(self.action_folder, str(self.game_id))
        os.mkdir(session_folder)
        bot_path = os.path.join(session_folder, self.BOTS_FOLDER_NAME)
        os.mkdir(bot_path)

        # Copy all the bot files into the action folder.
        lua_files = glob.glob(LUA_FILES_GLOB)
        assert len(lua_files) == 6
        for filename in lua_files:
            shutil.copy(filename, bot_path)

        # Finally, symlink DOTA to this folder.
        os.symlink(src=bot_path, dst=self.dota_bot_path)
        return bot_path

    async def monitor_log(self):
        while True:  # TODO(tzaman): probably just retry 10x sleep(0.5) then bust?
            filename = os.path.join(self.bot_path, self.CONSOLE_LOG_FILENAME)
            if os.path.exists(filename):
                with open(filename) as f:
                    for line in f:
                        # Demo line always comes before the LUADRY signal.
                        m_demo = self.RE_DEMO.search(line)
                        if m_demo and self.demo_path_rel is None:
                            self.demo_path_rel = m_demo.group(1)
                            print("(py) demo_path_rel='{}'".format(self.demo_path_rel))
                        m_luadry = self.RE_LUARDY.search(line)
                        if m_luadry:
                            config_json = m_luadry.group(1)
                            lua_config = json.loads(config_json)
                            print('(py) lua_config = ', lua_config)
                            self.lua_config_future.set_result(lua_config)
                            return
            await asyncio.sleep(0.2)

    async def run(self):
        # Start the worldstate listener(s).
        asyncio.create_task(self._run_dota())
        asyncio.create_task(self._worldstate_listener(port=self.PORT_WORLDSTATE_RADIANT))
        # asyncio.create_task(self.worldstate_listener(port=self.PORT_WORLDSTATE_DIRE))

    async def _run_dota(self):
        script_path = os.path.join(self.dota_path, self.DOTA_SCRIPT_FILENAME)
        GAME_MODE = 11
        # TODO(tzaman): all these options should be put in a proto and parsed with gRPC Config.
        args = [
            script_path,
            '-botworldstatesocket_threaded',
            '-botworldstatetosocket_dire {}'.format(self.PORT_WORLDSTATE_DIRE),
            '-botworldstatetosocket_frames {}'.format(self.ticks_per_observation),
            '-botworldstatetosocket_radiant {}'.format(self.PORT_WORLDSTATE_RADIANT),
            '-con_logfile scripts/vscripts/bots/{}'.format(self.CONSOLE_LOG_FILENAME),
            '-con_timestamp',
            '-console',
            '-insecure',
            '-noip',
            '-nowatchdog',  # WatchDog will quit the game if e.g. the lua api takes a few seconds.
            '+clientport 27006',  # Relates to steam client.
            '+dota_1v1_skip_strategy 1',
            '+dota_surrender_on_disconnect 0',
            '+host_timescale {}'.format(self.host_timescale),
            '+hostname dotaservice',
            '+map', 'start gamemode {}'.format(GAME_MODE),
            '+sv_cheats 1',
            '+sv_hibernate_when_empty 0',
            '+sv_lan 1',
            '+tv_delay 0 ',
            '+tv_enable 1',
            '+tv_title {}'.format(self.game_id),
            '+tv_autorecord 1',
            '+tv_transmitall 1',  # TODO(tzaman): what does this do exactly?
        ]
        if False:  # The viewer wants to play himself
            args.append('+dota_start_ai_game 1')
        else:
            args.append('-fill_with_bots')
        if not self.render:
            args.append('-dedicated')

        print('args=', args)

        create = asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            # stdout=asyncio.subprocess.PIPE,
            # stderr=asyncio.subprocess.PIPE,
        )
        self.process = await create

        # task_record_replay = asyncio.create_task(self.record_replay(process=self.process))
        task_monitor_log = asyncio.create_task(self.monitor_log())

        try:
            await self.process.wait()
        except asyncio.CancelledError:
            kill_processes_and_children(pid=self.process.pid)
            raise

    async def close(self):
        # TODO(tzaman): close async stuff?

        # Make the bot flush.
        self.write_action(data='FLUSH')

        # Stop the recording
        self.process.stdin.write(b"tv_stoprecord\n")
        self.process.stdin.write(b"quit\n")
        await self.process.stdin.drain()
        await asyncio.sleep(1)

        # Move the recording.
        if self.demo_path_rel is not None:
            demo_path_abs = os.path.join(self.dota_path, 'dota', self.demo_path_rel)
            try:
                shutil.move(demo_path_abs, self.bot_path)
            except Exception as e:  # Fail silently.
                print(e)


    @staticmethod
    async def _data_from_reader(reader):
        # Receive the package length.
        data = await reader.read(4)
        n_bytes = unpack("@I", data)[0]
        # Receive the payload given the length.
        # data = await asyncio.wait_for(reader.read(n_bytes), timeout=3.0)
        data = await reader.read(n_bytes) # Should we timeout for this?
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
                try:
                    parsed_data = await self._data_from_reader(reader)
                    # Only put data pre-game (4) and game (5).
                    # TODO(tzaman): use oficial enums from proto.
                    if parsed_data.game_state == 4 or parsed_data.game_state == 5:
                        self.worldstate_queue.put_nowait(parsed_data)
                except DecodeError as e:
                    print(e)
                    pass
        except asyncio.CancelledError:
            raise


class DotaService(DotaServiceBase):
    def __init__(self, dota_path, action_folder):
        self.dota_path = dota_path
        self.action_folder = action_folder

        # Initial assertions.
        verify_game_path(self.dota_path)

        if not os.path.exists(self.action_folder):
            if platform == "linux" or platform == "linux2":
                raise ValueError(
                    "Action folder '{}' not found.\nYou can create a 2GB ramdisk by executing:"
                    "`mkdir /tmpfs; mount -t tmpfs -o size=2048M tmpfs /tmpfs`\n"
                    "With Docker, you can add a tmpfs adding `--mount type=tmpfs,destination=/tmpfs`"
                    " to its run command.".format(self.action_folder))
            elif platform == "darwin":
                if not os.path.exists(self.action_folder):
                    raise ValueError(
                        "Action folder '{}' not found.\nYou can create a 2GB ramdisk by executing:"
                        " `diskutil erasevolume HFS+ 'ramdisk' `hdiutil attach -nomount ram://4194304``"
                        .format(self.action_folder))

        self.dota_game = None
        super().__init__()

    async def reset(self, stream):
        """reset method.

        This method should start up the dota game and the other required services.
        """
        print('DotaService::reset()')

        config = await stream.recv_message()
        print('config=\n', config)

        
        # Kill any previously running dota processes.
        # TODO(tzaman): Currently semi-gracefully. Can be cleaner.
        if self.dota_game is not None:
            await self.dota_game.close()
        os.system("ps | grep dota2 | awk '{print $1}' | xargs kill -9")

        # Create a new dota game instance.
        self.dota_game = DotaGame(
            dota_path=self.dota_path,
            action_folder=self.action_folder,
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
        self.dota_game.write_live_config(data=config)

        # Return the reponse
        await stream.send_message(Observation(world_state=data))

    async def step(self, stream):
        print('DotaService::step()')

        request = await stream.recv_message()

        action = MessageToDict(request.action)

        # Add the dotatime to the dict for verification.
        action['dota_time'] = self.dota_game.dota_time

        print('(python) action=', action)

        self.dota_game.write_action(data=action)

        # We've started to assume our queue will only have 1 item.
        data = await self.dota_game.worldstate_queue.get()

        # TODO(tzaman): I've seen empty worldstates on occasions. How to deal with that?

        # Update the tick
        self.dota_game.dota_time = data.dota_time

        # Make sure indeed the queue is empty and we're entirely in sync.
        assert self.dota_game.worldstate_queue.qsize() == 0

        # Return the reponse.
        await stream.send_message(Observation(world_state=data))


async def serve(server, *, host, port):
    await server.start(host, port)
    print('Serving on {}:{}'.format(host, port))
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()


async def grpc_main(loop, handler, host, port):
    server = Server([handler], loop=loop)
    await serve(server, host=host, port=port)


def main(grpc_host, grpc_port, dota_path, action_folder):
    dota_service = DotaService(dota_path=dota_path, action_folder=action_folder)
    loop = asyncio.get_event_loop()
    tasks = grpc_main(
        loop=loop,
        handler=dota_service,
        host=grpc_host,
        port=grpc_port,
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
