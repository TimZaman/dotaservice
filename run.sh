
# 4gb Ramdisk on OSX:
# diskutil erasevolume HFS+ 'ramdisk' `hdiutil attach -nomount ram://8388608`

# person that worked on the dota bot api at valve is "Chris Carollo"

docker run --net=host -it dotabot

/root/Steam/steamapps/common/dota\ 2\ beta/game/dota.sh \
    -dedicated -insecure +map dota +sv_lan 1 +clientport 27006 +sv_cheats 1 \
    -fill_with_bots -host_force_frametime_to_equal_tick_interval 1 \
    -botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121 -botworldstatetosocket_frames 5 -botworldstatesocket_threaded \
    +sv_hibernate_when_empty 0 +dota_auto_surrender_all_disconnected_timeout 180 +host_timescale 1


/root/Steam/steamapps/common/dota\ 2\ beta/game/dota.sh \
    -dedicated -insecure -console \
    +map dota +sv_cheats 1 +sv_lan 1 +clientport 27006 +hostname DotA_2

# ln -s /home/tzaman/.steam/steamcmd/linux32 /home/tzaman/.steam/sdk32
# ln -s /home/tzaman/.steam/steamcmd/linux64 /home/tzaman/.steam/sdk64

# On client, clientport default is '27015'. Multiple steam clients on same router/computer should
# have different clienports.
# +clientport 27017

# +sv_cheats 1

# 1v1 mid mode; so you can skip the 30s strategy.
# +dota_force_gamemode 21
# +dota_1v1_skip_strategy 1




# saving replays is impossibe by dedicated servers: need SourceTV (tv_*).
# 27020 is default sourcetv port. clients can just conncet with `connect ip:27020` and drop into the game
# /root/Steam/steamapps/common/dota\ 2\ beta/game/dota.sh -dedicated +map dota +sv_lan 1 +sv_hibernate_when_empty 0 +dota_auto_surrender_all_disconnected_timeout 180 +tv_enable 1 +tv_autorecord 1 +tv_dota_auto_record_stressbots 1 +tv_delay 0 +dota_force_gamemode 11 +dota_force_upload_match_stats 1

# Playing demo: `playdemo $demfile` looks in `dota 2 beta/game/dota` folder, not in `/replays`!
# Or you just do `playdemo replays/$demfile` and it'll work too.

# -condebug "logs all console output into the console.log text file."

# host_force_frametime_to_equal_tick_interval 1
# -startgame

# Set up the worldstate server.
#-botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121 -botworldstatetosocket_frames 10 -botworldstatesocket_threaded

#  -botworldstatetosocket_frames "to only send world state updates every N frames."" 

# Using this on desktop mac (12 nov):
# -novid -console -condebug -botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121 -botworldstatetosocket_frames 1 -botworldstatesocket_threaded -host_force_frametime_to_equal_tick_interval 1 +sv_cheats 1 +sv_lan 1 +dota_camera_edgemove 0