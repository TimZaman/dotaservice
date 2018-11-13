
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import asyncore
import socket
import time


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

    def recvall(self, buffer_size=4096):
        buf = self.recv(buffer_size)
        while buf:
            yield buf
            if len(buf) < buffer_size: break
            buf = self.recv(buffer_size)

    def handle_read(self):
        print('handle_read')

        response = b''.join(self.recvall())
        if not response:
            # Didn't receive anything. Can happen on bad connections.
            return

        print('response:', response)

        # self.handle_close() # we don't expect more data

    def handle_close(self):
        self.close()



HOST = '127.0.0.1'
PORT_RADIANT = 12120
PORT_DIRE = 12121



RETRY_DELAY = 5

while True:
    try:
        print('loopin..')
        radiant_dispatcher = Observatory(host=HOST, port=PORT_RADIANT)
        dire_dispatcher = Observatory(host=HOST, port=PORT_DIRE)
        asyncore.loop()
    except ConnectionRefusedError:
        del radiant_dispatcher, dire_dispatcher
        print('Connection refused. Trying again in {} seconds.'.format(RETRY_DELAY))
        time.sleep(RETRY_DELAY)
