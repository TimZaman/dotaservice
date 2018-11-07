
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify, abort
import time
from collections import namedtuple
import json
import os

app = Flask(__name__)


ACTION_FOLDER = '/Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game/dota/scripts/vscripts/bots/actions/'
# ACTION_FOLDER = '/Volumes/RAM Disk/'

class Action(object):

    def __init__(self, tick, move_x, move_y):
        self.tick = tick
        self.move_x = move_x
        self.move_y = move_y
    
    def to_dict(self):
        return {'tick': self.tick, 'move_x': self.move_x, 'move_y': self.move_y}

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_lua_file(self, filename):
        """Writing to an imporable lua file in json format."""
        json = self.to_json()
        data = "data = '{}'".format(json)
        print(data)
        with open(filename, 'w') as f:
            f.write(data)


def get_action(tick):
    """Get the action for the given tick."""
    return Action(move_x=1337, move_y=666, tick=tick)




@app.route(r"/action", methods=['POST'])
def post():
    game_id  = request.args.get('game_id')
    tick  = request.args.get('tick')
    print('post(game_id=%s, tick=%s)...' % (game_id, tick))
    print('hi')
    response = {}
    response['status'] = 200
    response['tick'] = tick
    data = request.get_json()
    if data == None:
        # this should raise an HTTPException
        abort(400, 'POST Data was not JSON')
    else:
        print('data: %s' % data)

    action = get_action(tick)
    filename = os.path.join(ACTION_FOLDER, game_id, '{}.lua'.format(tick))
    print('filename', filename)
    action.to_lua_file(filename)

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000


    post('my_game_id', -1300)