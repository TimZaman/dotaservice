This folder includes my effort of making a C++ based bot.
Compiling the files into libraries called `botcpp_dire.so` and
`botcpp_radiant.so` and putting them into Dota's `vscripts/bots`
directory will have them loaded using `dlopen` by Dota if you
are playing with bots.

You can compile the enclosed `.cpp` files using:

```sh
g++ -shared -o botcpp_dire.so -fPIC botcpp_dire.cpp
```

The symbols that are required are `Init`, `Observe`, `Act` and
`Shutdown`, but I do not know their signatures. `Init` will 
actually execute, but I never got any invocations of the other
symbols. I didn't have time, and it might not be allowed, to
reverse engineer anything, so I didn't. I did send Gabe Newell
an email asking for help exposing the C++ API (seriously).
