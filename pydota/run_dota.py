
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify, abort
import subprocess
import time
import urllib
from urllib import request
import threading
import os
import signal
import subprocess
import atexit
import psutil


app = Flask(__name__)

p = None  # Object to hold the subprocess


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


def kill_p():
    """Kill all children of the process recursively."""
    kill_child_processes(p.pid)


@app.route(r"/run", methods=['POST'])
def run():
    global p
    args = [
        "/Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game/dota.sh",
        "-dedicated",
        "-insecure",
        "-console",
        # "+map dota",
    ]
    with open("stdout.txt", "wb") as out, open("stderr.txt", "wb") as err:
        p = subprocess.Popen(args,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            # preexec_fn=os.setpgrp
            )
    atexit.register(kill_p)
    return ''


@app.route(r"/write", methods=['POST'])
def write():
    p.stdin.write(b"HAHAHA\n")
    p.stdin.flush()
    return ''


if __name__ == '__main__':
    app.run(port=5000) #run app in debug mode on port 5000

