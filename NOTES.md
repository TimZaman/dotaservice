Below are random notes I am taking during development. I hope you like spaghetti!

# Releasing to PyPI

1. Change version info in `setup.py` and root `__init__.py` and commit.
2. Add tag `git tag -a $version $sha` and push `git push --tags`
3. Build release `python3 setup.py sdist bdist_wheel`.
4. Upload to `pypi`: `twine upload --repository-url https://upload.pypi.org/legacy/ dist/*`.

# ETC

Adding things to the Dota bot API's `package.path` in LUA doesn't seem to work. It refuses to
mount anything outside the `vscripts` directory. The solution is to mount your external disk simply
by linking the folder e.g.
```sh
ln -s /Volumes/ramdisk/ /Users/tzaman/Library/Application Support/Steam/SteamApps/common/dota 2 beta/game/dota/scripts/vscripts
```


Check open sockets:
```sh
$ lsof -n -i | grep -e LISTEN -e ESTABLISHED
>>> dota2     1061 tzaman  215u  IPv4 0x84b2d9c103958a77      0t0  TCP *:12120 (LISTEN)
```

DOTA comes with TensorFlow libs (on ubuntu gives `failed to dlopen libtensorflow.so error=libtensorflow.so: cannot open shared object file: No such file or directory`), primarily used for dota plus features:

```sh
$ cat Steam/steamapps/common/dota\ 2\ beta/game/dota/bin/linuxsteamrt64/libserver.so | grep -a tensorflow
>>> tf_server_client_connect_timeout_stf_server_client_read_timeout_stf_server_client_write_timeout_stf_server_stats_spew_interval_sdota_suggest_spew_pregame_itemsdota_suggest_spew_win_probabilitydota_suggest_spew_win_probability_chatdota_suggest_pregame_items_reductiondota_suggest_pregame_items_thresholddota_suggest_item_sequence_allow_thresholddota_suggest_item_sequence_threshold_startdota_suggest_item_sequence_threshold_fulldota_suggest_item_sequence_other_option_multiplierdota_suggest_item_sequence_dupe_multiplierdota_suggest_lane_trilane_penaltydota_suggest_win_probability_interval
```

The bot script's LUA version is 5.1 and luajit-2.0.4.

Good command for dedicated bot 
```sh
dota.sh \
-dedicated -insecure +map dota +sv_lan 1 +clientport 27006 +sv_cheats 1 \
-fill_with_bots \
-botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121 -botworldstatetosocket_frames 5 -botworldstatesocket_threaded \
+sv_hibernate_when_empty 0 +dota_auto_surrender_all_disconnected_timeout 180 +host_timescale 1 \
+dota_camera_edgemove 0
```

Clear dangling images.
```sh
docker rmi $(docker images -f dangling=true -q)
```

Create a 4GB Ramdisk on MacOS
```sh
diskutil erasevolume HFS+ 'ramdisk' `hdiutil attach -nomount ram://8388608`
```

The awesome Valve dev that worked on the dota bot api and OpenAI's contact at Valve is _Chris Carollo_.

On the DOTA client, `clientport` default is `27015`. Multiple steam clients on same router/computer should
have different clienports; corresponding option e.g.: `+clientport 27017`.

Mid 1v1 mode skips the 30s strategy; `+dota_force_gamemode 11 +dota_1v1_skip_strategy 1`.

Saving replays is impossibe by dedicated servers: need SourceTV (tv_*), `27020` is the default SourceTV port.
Clients can just connect with console command `connect ip:27020` and drop into the game.
Notice SourceTV is considered a Player and BOT, and usually gets the first index (1).

```sh
# /root/Steam/steamapps/common/dota\ 2\ beta/game/dota.sh -dedicated +map dota +sv_lan 1 +sv_hibernate_when_empty 0 +dota_auto_surrender_all_disconnected_timeout 180 +tv_enable 1 +tv_autorecord 1 +tv_dota_auto_record_stressbots 1 +tv_delay 0 +dota_force_gamemode 11 +dota_force_upload_match_stats 1
```
Playing demo: `playdemo $demfile` looks in `dota 2 beta/game/dota` folder, not in `/replays`!
Or you just do `playdemo replays/$demfile` and it'll work too.
Playing the auto-replay of GUI games (bot vs bot, or with human added) works. The auto-replay
of dedicated games seems to result in a fully black screen.
When you start a game with the GUI, the map is called `+map start`. If you have a dedicated
server running with `+map dota`, its replay stays black and doesn't seem to work. If you start it
with `+map start`, it will load with some errors, which are circumvented if you scroll to a certain
tick in the replay file.

What is the difference between game mod 11 (Mid Only) and 21 (Solo Mid 1 vs 1)?

With a speed of 300, it takes around 30 seconds to walk from spawn (fountain) to mid ~(0,0)

A mid game without heroes can already be over in 15 minutes by creeps alone taking the tower.

Even with `host_force_frametime_to_equal_tick_interval` set, there is a bug in dotatime, as there
is an extra tick inserted at 0; e.g.: `[-0.034439086914062, -0.0011062622070312, 0, 0.033332824707031]`.
The corresponding gamestates are `[4, 4, 5, 5]`, so the tick at exactly 0 is the `DOTA_GAMERULES_STATE_GAME_IN_PROGRESS`.
This is also a problem with the botworldstate; e.g. if you have a state every n frames, when it wraps
around zero time, it will add another tick there.

The game is trying to load C++ bot libraries using
dlopen with mode 6 (`RTLD_LOCAL | RTLD_NOW`); `bots/botcpp_radiant.so` and `bots/botcpp_dire.so`.
It expects (for what i found using `LD_DEBUG=bindings LD_BIND_NOW=1`) an implemetation of `Init`,
`Observe`, `Act` and `Shutdown`.

Dota's worldstate server will clog up quickly with a host timescale of 10 and printing out every state
tick.

The worldstate can be set to be pushed every n ticks (`botworldstatetosocket_frames`) but the
tick at which it starts pushing is not deterministic.

The worldstate server starts even before the gameplay starts and bot Think's begin;
E.g. the states roll in at  dotatime `-75` (probably state `DOTA_GAMERULES_STATE_HERO_SELECTION`);
then jumps to `-59.97` after entering state `DOTA_GAMERULES_STATE_WAIT_FOR_MAP_TO_LOAD`.
Once it enters state 'DOTA_GAMERULES_STATE_PRE_GAME' it resets to dotatime `-90`.

The bot scripts are reloaded entirely with every game, not just once when booting dota.

The `CreateRemoteHTTPRequest` from the LUA api does not work when steam is off on my mac OS-X.
If I run in offline mode and/or take out my internet cable it works, but once the Steam client is
turned off, the HTTP request does not work anymore (returns `nil`). I wonder what the stability of
this part of the API is, should maybe avoid using.


The tick-to-dotatime relation is inaccurate, ticks aren't given in the worldstate too. Probably
an issue with float/double etc. So move away from ticks, and use time as %.2f integers, or time*30,
or just cut off until %.2f and then floor. On an hour long game, there were 29.981 ticks/s, which
makes me thing the ticks and s are unreliable, even with `host_force_frametime_to_equal_tick_interval` on,
which doesn't seem to help at all.

All the output from the lua console isn't flushed upon redicretion in python (or piping or tee).

The official proto for the botscript can be found here:
https://github.com/SteamDatabase/Protobufs/blob/master/dota2/dota_gcmessages_common_bot_script.proto

Source has options for streaming logs using udp `-log on` but none of that actually seems to work
with dota.

The `restart` command restarts the game. The `quit` command exits it entirely. `restart` is really
fast but it will complicate resetting many states. Q: Are all the botscripts called again?

When watching a 5v5 bot game live, you will have full for of war because you're not on a team,
so you need to do `jointeam spec` to see the map.

When running without `-dedicated`, the console doesn't show VSript (lua) stdout/stderr anymore.

Auto-saving replays might be a good way to go as it records everything. I didn't find any options
how to specify where to save the auto-replay. A workaround it to use a `record $filename` command
in console.

Running dota headless only takes up around 250Mb RAM (woah that's little!).

At dotatime=35s is where the first creeps get killed (by natural causes), so this is when xp can be
gained.

Something is up with the host timescale, it sometimes runs 2x faster than it should. E.g. when
you `render` the bothmatch, and you focus on the screen it goes at 2x the speed. If you then alt-tab
to another window, it suddenly goes at normal speed. Why? Bisect this issue by toggling some
dota launch variables. Seems to be related to `host_force_frametime_to_equal_tick_interval`?

Investigate again the difference between the maps `start` and `dota`?

The dota game map spans from -7000:7000 and the 0,0 point is just outside the dire tower's range
near the river, on the radiant high ground.

In the Dota lua api `Action_MoveToLocation` is quantized and doesn't move small distances (<250 units)

You cannot have the dota GUI when steam isn't running. Running dedicated server witout steam is 
fine, but so you cannot `render` with `True` when Steam isn't running, or Dota will quit
'unexpectedly'.


`./dota.sh -console -insecure +sv_cheats 1 +map dota +dota_start_ai_game 1`
Starts a game where the player can pick the hero it wants and the rest are filled in with bots.

`./dota.sh -console -insecure +sv_cheats 1 +map dota +dota_start_ai_game 1 +dota_force_gamemode 11`
Seems the same as previous.

Changing above to `+map start` doesn't make a difference.


`./dota.sh -console -insecure +sv_lan 1 +sv_cheats 1 +map "dota gamemode 21" -fill_with_bots`
Works nicely. Not sure how to take over bots though. Seems that i can just join a team as long
as I remove `-fill_with_bots` and then use `+dota_start_ai_game 1` instead:
`./dota.sh -console -insecure +sv_lan 1 +sv_cheats 1 +map "dota gamemode 21" +dota_start_ai_game 1`
You can choose a side with `cl_team` (2=radiant, 3=dire)
Then you can choose a side and the rest will be fille din with bots. Would this still work with
dedicated game?

`dota_lobby_browser_selected_gamemode`?

`dota_bot_use_machine_learned_weights` is a console option?

The difference between game mode MID (11) and 1v1 (21) seems that in the mid game mode, the creeps
spawn at the -60s time (after 30s), while at 21 they spawn at 0s (after 90s).

So `dota_start_ai_game` auto-fills with bots after the player has chosen. Does it work with
no real players still? -> NO. So if playing through GUI, use `dota_start_ai_game` otherwise use
`-fill_with_bots`

Demos often seem borked because of abrupt quits. Just quit the dota game cleanly and the demos will be
allright, even with `map` is dota (and gamemode 11).

You can also launch a demo from console simply with `./dota.sh +playdemo foo` if `foo.dem` is the demo.

Confirmed, map `start` works fine with replay (even if steam is on or off) while map `dota` gives
a black screen on the replays.

When using the GUI, the terminal's stdin doesn't pipe to dota's console anymore.

Maybe we can use `host_force_frametime_to_equal_tick_interval` and `host_framerate` to speed things up.

Maybe use these:
* dota_bot_use_machine_learned_weights 
* dota_spectator_auto_spectate_bot_games 
* dota_override_dire_team_logo
* mp_defaultteam
* cl_team

Dota sends out the dire and radiant worldstates out in below order. This is quite problematic
because a radiant move always needs to be made, even before the dire has received their observation.
1) Send out radiant worldstate
2) Invoke all lua scripts for radiant team
3) Send out dire worldstate
4) Invoke all lua scripts for radiant team
5) repeat (1)

You can run a custom lua game using:
```./dota.sh +dota_launch_custom_game hero_demo hero_demo_main```
These addons are in `game/dota_addons/...`. However, workshop tools for maps and such
require Windows OS. It seems that whatever you try, the worldstates are not being sent out correctly,
or they are empty. You can set them up, but it seems they just contain no valid data..
