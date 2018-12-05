from sys import platform
import argparse
import os

from dotaservice.dotaservice import main
from dotaservice.dotaservice import verify_game_path


def get_default_game_path():
    game_path = None
    if platform == "linux" or platform == "linux2":
        game_path = os.path.expanduser("~/Steam/steamapps/common/dota 2 beta/game")
    elif platform == "darwin":
        game_path = os.path.expanduser(
            "~/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game")
    return game_path


def get_default_action_path():
    action_path = None
    if platform == "linux" or platform == "linux2":
        action_path = "/tmp/"
    elif platform == "darwin":
        action_path = "/Volumes/ramdisk/"
    return action_path


if platform not in ["linux", "linux2", "darwin"]:
    raise EnvironmentError("Platform {} not supported.".format(platform))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ip", type=str, help="gRPC host ip", default="")
parser.add_argument("--port", type=int, help="gRPC port", default=13337)
parser.add_argument(
    "--game-path",
    type=str,
    default=get_default_game_path(),
    help="Path to the dota entrypoint (dota.sh).")
parser.add_argument(
    "--action-path",
    type=str,
    default=get_default_action_path(),
    help="Path to the root folder in which the game logs will be saved.")
args = parser.parse_args()

main(
    grpc_host=args.ip,
    grpc_port=args.port,
    dota_path=args.game_path,
    action_folder=args.action_path,
)
