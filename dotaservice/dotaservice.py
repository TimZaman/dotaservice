from pprint import pprint, pformat
from struct import unpack
from sys import platform
import asyncio
import glob
import json
import logging
import math
import os
import pkg_resources
import re
import shutil
import signal
import time
import traceback
import uuid

from google.protobuf.message import DecodeError
from google.protobuf.json_format import MessageToDict
from grpclib.server import Server

from dotaservice import __version__
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMERULES_STATE_GAME_IN_PROGRESS
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMERULES_STATE_PRE_GAME
from dotaservice.protos.DotaService_grpc import DotaServiceBase
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import HOST_MODE_DEDICATED, HOST_MODE_GUI, HOST_MODE_GUI_MENU
from dotaservice.protos.DotaService_pb2 import TEAM_RADIANT, TEAM_DIRE
from dotaservice.protos.DotaService_pb2 import Observation, ObserveConfig, InitialObservation
from dotaservice.protos.DotaService_pb2 import Status

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

LUA_FILES_GLOB = pkg_resources.resource_filename('dotaservice', 'lua/*.lua')
LUA_FILES_GLOB_ACTIONS = pkg_resources.resource_filename('dotaservice', 'lua/actions/*.lua')


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

    ACTIONS_FILENAME_FMT = 'actions_t{team_id}'
    ACTIONABLE_GAME_STATES = [DOTA_GAMERULES_STATE_PRE_GAME, DOTA_GAMERULES_STATE_GAME_IN_PROGRESS]
    BOTS_FOLDER_NAME = 'bots'
    CONFIG_FILENAME = 'config_auto'
    CONSOLE_LOG_FILENAME = 'console.log'
    CONSOLE_LOGS_GLOB = 'console*.log'
    DOTA_SCRIPT_FILENAME = 'dota.sh'
    LIVE_CONFIG_FILENAME = 'live_config_auto'
    PORT_WORLDSTATES = {TEAM_RADIANT: 12120, TEAM_DIRE: 12121} 
    RE_DEMO =  re.compile(r'playdemo[ \t](.*dem)')
    RE_LUARDY = re.compile(r'LUARDY[ \t](\{.*\})')
    WORLDSTATE_PAYLOAD_BYTES = 4

    def __init__(self,
                 dota_path,
                 action_folder,
                 remove_logs,
                 host_timescale,
                 ticks_per_observation,
                 game_mode,
                 host_mode,
                 game_id=None,
                 ):
        self.dota_path = dota_path
        self.action_folder = action_folder
        self.remove_logs = remove_logs
        self.host_timescale = host_timescale
        self.ticks_per_observation = ticks_per_observation
        self.game_mode = game_mode
        self.host_mode = host_mode
        self.game_id = game_id
        if not self.game_id:
            self.game_id = str(uuid.uuid1())
        self.dota_bot_path = os.path.join(self.dota_path, 'dota', 'scripts', 'vscripts',
                                          self.BOTS_FOLDER_NAME)
        self.bot_path = self._create_bot_path()
        self.worldstate_queues = {
            TEAM_RADIANT: asyncio.Queue(loop=asyncio.get_event_loop()),
            TEAM_DIRE: asyncio.Queue(loop=asyncio.get_event_loop()),
        }
        self.lua_config_future = asyncio.get_event_loop().create_future()
        self._write_config()
        self.process = None
        self.demo_path_rel = None

        if self.host_mode != HOST_MODE_DEDICATED:
            # TODO(tzaman): Change the proto so that there are per-hostmode settings?
            self.host_timescale = 1

        has_display = 'DISPLAY' in os.environ or platform == "darwin"
        if not has_display and host_mode != HOST_MODE_DEDICATED:
            raise ValueError('GUI requested but no display detected.')
            exit(-1)
        super().__init__()

    def _write_config(self):
        # Write out the game configuration.
        config = {
            'game_id': self.game_id,
            'ticks_per_observation': self.ticks_per_observation,
        }
        self.write_static_config(data=config)

    def write_static_config(self, data):
        self._write_bot_data_file(filename_stem=self.CONFIG_FILENAME, data=data)

    def write_live_config(self, data):
        logger.debug('Writing live_config={}'.format(data))
        self._write_bot_data_file(filename_stem=self.LIVE_CONFIG_FILENAME, data=data)
        
    def write_action(self, data, team_id):
        filename_stem = self.ACTIONS_FILENAME_FMT.format(team_id=team_id)
        self._write_bot_data_file(filename_stem=filename_stem, data=data)

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
        for filename in lua_files:
            shutil.copy(filename, bot_path)

        # Copy all the bot action files into the actions subdirectory
        action_path = os.path.join(bot_path, "actions")
        os.mkdir(action_path)
        action_files = glob.glob(LUA_FILES_GLOB_ACTIONS)
        for filename in action_files:
            shutil.copy(filename, action_path)

        # shutil.copy('/Users/tzaman/dev/dotaservice/botcpp/botcpp_radiant.so', bot_path)

        # Finally, symlink DOTA to this folder.
        os.symlink(src=bot_path, dst=self.dota_bot_path)
        return bot_path

    async def monitor_log(self):
        abs_glob = os.path.join(self.bot_path, self.CONSOLE_LOGS_GLOB)
        while True:
            # Console logs can get split from `$stem.log` into `$stem.$number.log`.
            for filename in glob.glob(abs_glob):
                with open(filename) as f:
                    for line in f:
                        # Demo line always comes before the LUADRY signal.
                        if self.demo_path_rel is None:
                            m_demo = self.RE_DEMO.search(line)
                            if m_demo:
                                self.demo_path_rel = m_demo.group(1)
                                logger.debug("demo_path_rel={}".format(self.demo_path_rel))
                        m_luadry = self.RE_LUARDY.search(line)
                        if m_luadry:
                            config_json = m_luadry.group(1)
                            lua_config = json.loads(config_json)
                            logger.debug('lua_config={}'.format(lua_config))
                            self.lua_config_future.set_result(lua_config)
                            return
            await asyncio.sleep(0.2)

    async def run(self):
        asyncio.create_task(self._run_dota())
        # Start the worldstate listener(s).
        for team_id in self.worldstate_queues:
            asyncio.create_task(self._worldstate_listener(
                port=self.PORT_WORLDSTATES[team_id], queue=self.worldstate_queues[team_id], team_id=team_id))

    async def _run_dota(self):
        script_path = os.path.join(self.dota_path, self.DOTA_SCRIPT_FILENAME)

        # TODO(tzaman): all these options should be put in a proto and parsed with gRPC Config.
        args = [
            script_path,
            '-botworldstatesocket_threaded',
            '-botworldstatetosocket_frames {}'.format(self.ticks_per_observation),
            '-botworldstatetosocket_radiant {}'.format(self.PORT_WORLDSTATES[TEAM_RADIANT]),
            '-botworldstatetosocket_dire {}'.format(self.PORT_WORLDSTATES[TEAM_DIRE]),
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
            '+sv_cheats 1',
            '+sv_hibernate_when_empty 0',
            '+tv_delay 0 ',
            '+tv_enable 1',
            '+tv_title {}'.format(self.game_id),
            '+tv_autorecord 1',
            '+tv_transmitall 1',  # TODO(tzaman): what does this do exactly?
        ]

        if self.host_mode == HOST_MODE_DEDICATED:
            args.append('-dedicated')
        if self.host_mode == HOST_MODE_DEDICATED or \
            self.host_mode == HOST_MODE_GUI:
            args.append('-fill_with_bots')
            args.extend(['+map', 'start gamemode {}'.format(self.game_mode)])
            args.append('+sv_lan 1')
        if self.host_mode == HOST_MODE_GUI_MENU:
            args.append('+sv_lan 0')

        # Supress stdout if the logger level is info.
        stdout = None if logger.level == 'INFO' else asyncio.subprocess.PIPE

        create = asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE, stdout=stdout, stderr=stdout,
        )
        self.process = await create

        task_monitor_log = asyncio.create_task(self.monitor_log())

        try:
            await self.process.wait()
        except asyncio.CancelledError:
            kill_processes_and_children(pid=self.process.pid)
            raise


    def _move_recording(self):
        logger.info('::_move_recording')
        # Move the recording.
        # TODO(tzaman): high-level: make recordings optional.
        # TODO(tzaman): retain discarded recordings:
        #   EXAMPLE FROM CONSOLE:
        #   EX: "Discarding replay replays/auto-20181228-2311-start-dotaservice.dem"
        #   EX: "Renamed replay replays/auto-20181228-2311-start-dotaservice.dem to replays/discarded/replays/auto-20181228-2311-start-dotaservice.dem"
        if self.demo_path_rel is not None:
            demo_path_abs = os.path.join(self.dota_path, 'dota', self.demo_path_rel)
            try:
                shutil.move(demo_path_abs, self.bot_path)
            except Exception as e:  # Fail silently.
                logger.error(e)

    async def close(self):
        logger.info('::close')

        # If the process still exists, clean up.
        if self.process.returncode is None:
            logger.debug('flushing bot')
            # Make the bot flush.
            for team_id in [TEAM_RADIANT, TEAM_DIRE]:
                self.write_action(data='FLUSH', team_id=team_id)
            # Stop and move the recording
            logger.debug('stopping recording')
            self.process.stdin.write(b"tv_stoprecord\n")
            self.process.stdin.write(b"quit\n")
            await self.process.stdin.drain()
            await asyncio.sleep(1)

        self._move_recording()

        if self.remove_logs:
            shutil.rmtree(self.bot_path, ignore_errors=True)


    @classmethod
    async def _world_state_from_reader(cls, reader, team_id):
        # Receive the package length.
        data = await reader.read(cls.WORLDSTATE_PAYLOAD_BYTES)
        if len(data) != cls.WORLDSTATE_PAYLOAD_BYTES:
            # raise ValueError('Invalid worldstate payload')
            return None
        n_bytes = unpack("@I", data)[0]
        # Receive the payload given the length.
        data = await asyncio.wait_for(reader.read(n_bytes), timeout=2)
        # Decode the payload.
        world_state = CMsgBotWorldState()
        world_state.ParseFromString(data)
        logger.debug('Received world_state: dotatime={}, gamestate={}, team={}'.format(
            world_state.dota_time, world_state.game_state, team_id))
        return world_state

    @classmethod
    async def _worldstate_listener(self, port, queue, team_id):
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
                # This reader is always going to need to keep going to keep the buffers flushed.
                try:
                    world_state = await self._world_state_from_reader(reader, team_id)
                    if world_state is None:
                        logger.debug('Finishing worldstate listener (team_id={})'.format(team_id))
                        return
                    is_in_game = world_state.game_state in self.ACTIONABLE_GAME_STATES
                    has_units = len(world_state.units) > 0
                    if is_in_game and has_units:
                        # Only regard worldstates that are actionable (in-game + has units).
                        queue.put_nowait(world_state)
                except DecodeError as e:
                    pass
        except asyncio.CancelledError:
            raise


class DotaService(DotaServiceBase):

    def __init__(self, dota_path, action_folder, remove_logs):
        self.dota_path = dota_path
        self.action_folder = action_folder
        self.remove_logs = remove_logs

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

    @property
    def observe_timeout(self):
        if self.dota_game.host_mode == HOST_MODE_DEDICATED:
            return 10
        return 3600

    @staticmethod
    def stop_dota_pids():
        """Stop all dota processes.
        
        Stopping dota is nessecary because only one client can be active at a time. So we clean
        up anything that already existed earlier, or a (hanging) mess we might have created.
        """
        dota_pids = str.split(os.popen("ps -e | grep dota2 | awk '{print $1}'").read())
        for pid in dota_pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def clean_resources(self):
        """Clean resoruces.
        
        Kill any previously running dota processes, and therefore set our status to ready.
        """
        # TODO(tzaman): Currently semi-gracefully. Can be cleaner.
        if self.dota_game is not None:
            # await self.dota_game.close()
            self.dota_game = None
        self.stop_dota_pids()

    async def reset(self, stream):
        """reset method.

        This method should start up the dota game and the other required services.
        """
        logger.info('DotaService::reset()')

        config = await stream.recv_message()
        logger.debug('config=\n{}'.format(config))

        await self.clean_resources()

        # Create a new dota game instance.
        self.dota_game = DotaGame(
            dota_path=self.dota_path,
            action_folder=self.action_folder,
            remove_logs=self.remove_logs,
            host_timescale=config.host_timescale,
            ticks_per_observation=config.ticks_per_observation,
            game_mode=config.game_mode,
            host_mode=config.host_mode,
            game_id=config.game_id,
        )

        # Start dota.
        asyncio.create_task(self.dota_game.run())

        # We first wait for the lua config. TODO(tzaman): do this in DotaGame?
        logger.debug('::reset is awaiting lua config..')
        lua_config = await self.dota_game.lua_config_future
        logger.debug('::reset: lua config received={}'.format(lua_config))

        # Cycle through the queue until its empty, then only using the latest worldstate.
        data = {TEAM_RADIANT: None, TEAM_DIRE: None}
        for team_id in self.dota_game.worldstate_queues:
            try:
                while True:
                    # Deplete the queue.
                    queue = self.dota_game.worldstate_queues[team_id]
                    data[team_id] = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

        assert data[TEAM_RADIANT] is not None
        assert data[TEAM_DIRE] is not None

        if data[TEAM_RADIANT].dota_time != data[TEAM_DIRE].dota_time:
            raise ValueError(
                'dota_time discrepancy in depleting initial worldstate queue.\n'
                'radiant={:.2f}, dire={:.2f}'.format(data[TEAM_RADIANT].dota_time, data[TEAM_DIRE].dota_time))

        last_dota_time = data[TEAM_RADIANT].dota_time

        # Now write the calibration file.
        config = {
            'calibration_dota_time': last_dota_time
        }
        self.dota_game.write_live_config(data=config)

        # Return the reponse
        await stream.send_message(InitialObservation(
            world_state_radiant=data[TEAM_RADIANT],
            world_state_dire=data[TEAM_DIRE],
        ))


    async def observe(self, stream):
        logger.debug('DotaService::observe()')

        request = await stream.recv_message()
        team_id = request.team_id

        queue = self.dota_game.worldstate_queues[team_id]

        try:
            data = await asyncio.wait_for(queue.get(), timeout=self.observe_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # A timeout probably means the game is done
            # TODO(tzaman): how does one know when a game is finished?
            await stream.send_message(Observation(
                status=Status.Value('RESOURCE_EXHAUSTED'),
                team_id=team_id,
                ))
            return     

        # Make sure indeed the queue is empty and we're entirely in sync.
        assert queue.qsize() == 0

        # Return the reponse.
        await stream.send_message(Observation(
            status=Status.Value('OK'),
            world_state=data,
            team_id=team_id,
            ))

    async def act(self, stream):
        logger.debug('DotaService::act()')

        request = await stream.recv_message()
        team_id = request.team_id
        actions = MessageToDict(request.actions)

        logger.debug('team_id={}, actions=\n{}'.format(team_id, pformat(actions)))

        self.dota_game.write_action(data=actions, team_id=team_id)

        # Return the reponse.
        await stream.send_message(Empty())




async def serve(server, *, host, port):
    await server.start(host, port)
    logger.info('DotaService {} serving on {}:{}'.format(__version__, host, port))
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()


async def grpc_main(loop, handler, host, port):
    server = Server([handler], loop=loop)
    await serve(server, host=host, port=port)


def main(grpc_host, grpc_port, dota_path, action_folder, remove_logs, log_level):
    logger.setLevel(log_level)
    dota_service = DotaService(
        dota_path=dota_path,
        action_folder=action_folder,
        remove_logs=remove_logs,
        )
    loop = asyncio.get_event_loop()
    tasks = grpc_main(
        loop=loop,
        handler=dota_service,
        host=grpc_host,
        port=grpc_port,
    )

    loop.run_until_complete(tasks)
