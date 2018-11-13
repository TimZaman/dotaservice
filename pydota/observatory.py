
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import asyncore
import socket
from struct import unpack
import time
import math

import protobuf.CMsgBotWorldState_pb2 as _pb
import google.protobuf.text_format as txtf

TICKS_PER_SECOND = 30

def dotatime_to_tick(dotatime):
    return math.floor(dotatime * TICKS_PER_SECOND)


class Observatory(asyncore.dispatcher):

    RETRY_DELAY = 5

    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.host = host
        self.port = port
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((self.host, self.port))

    def handle_error(self):
        raise ConnectionRefusedError()
        self.handle_close()

    def writable(self):
        return 0 # don't have anything to write

    def handle_connect(self):
        pass # connection succeeded

    def handle_expt(self):
        self.close() # connection failed, shutdown

    def handle_read(self):
        print('handle_read')

        n_bytes = unpack("@I", self.recv(4))[0]
    
        data = self.recv(n_bytes)

        parsed_data = _pb.CMsgBotWorldState()
        parsed_data.ParseFromString(data)
        dotatime = parsed_data.dota_time
        tick = dotatime_to_tick(dotatime)
        print('tick: {}'.format(tick))

        # self.handle_close()

    def handle_close(self):
        self.close()



HOST = '127.0.0.1'
PORT_RADIANT = 12120
PORT_DIRE = 12121


RETRY_DELAY = 5

# while True:
#     try:
print('loopin..')
radiant_dispatcher = Observatory(host=HOST, port=PORT_RADIANT)
# dire_dispatcher = Observatory(host=HOST, port=PORT_DIRE)
asyncore.loop()
    # except ConnectionRefusedError:
    #     del radiant_dispatcher#, dire_dispatcher
    #     print('Connection refused. Trying again in {} seconds.'.format(RETRY_DELAY))
    #     time.sleep(RETRY_DELAY)
