This folder includes my effort of making a C++ based bot.
Compiling the files into libraries called `botcpp_dire.so` and
`botcpp_radiant.so` and putting them into Dota's `vscripts/bots`
directory will have them loaded using `dlopen` by Dota if you
are playing with bots.

You can compile the enclosed `.cpp` files using:

```sh
g++ -shared -o botcpp_radiant.so -fPIC botcpp_radiant.cpp dota_gcmessages_common_bot_script.pb.cc -std=c++11 -lprotobuf
```

The symbols that are required are `Init`, `Observe`, `Act` and
`Shutdown`, but I do not know their signatures. `Init` will 
actually execute, but I never got any invocations of the other
symbols. I didn't have time, and it might not be allowed, to
reverse engineer anything, so I didn't. I did send Gabe Newell
an email asking for help exposing the C++ API (seriously).

In order for above `Act` and `Observe` to be invoked, your bot world state sockets need to be
open and flushing. Those functions are called whenever the world state socket is being published.
For the flushing, you can just use the `world_state_listener.py` script.
See the `cpp` source files for more clarification.