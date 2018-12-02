local dkjson = require( "game/dkjson" )
local config = require("bots/config")

local LIVE_CONFIG_FILENAME = 'bots/live_config_auto'
local ACTION_FILENAME = 'bots/action'
local live_config = nil


local function dotatime_to_ms(dotatime)
    return string.format("%09d", math.floor(dotatime*1000))
end


local function act(action)
    print('(lua) act')
    local npcBot = GetBot()
    npcBot:Action_MoveToLocation(Vector(0, 0))
end


local function get_new_action(time_ms)
    print('(lua) get_new_action ', time_ms)
    local file_fn = nil
    while true do
        -- Try to load the file first
        while true do
            file_fn = loadfile(ACTION_FILENAME)
            if file_fn ~= nil then break end
        end
        -- Execute the file_fn; this loads contents into `data`.
        local data = file_fn()
        if data ~= nil then
            local data, pos, err = dkjson.decode(data, 1, nil)
            if err then
                print("(lua) JSON Decode Error=", err, " at pos=", pos)
            end
            if data.dota_time == time_ms then
                return data
            end
        end
    end
end


local function data_from_file(filename)
    -- Get the response from a file, while waiting for the file.
    print('(lua) looking for loadfile =', filename)
    local file_fn = nil
    while true do
        file_fn = loadfile(filename)
        if file_fn ~= nil then break end
    end
    -- Execute the file_fn; this loads contents into `data`.
    local data = file_fn()
    local data, pos, err = dkjson.decode(data, 1, nil)
    if err then
        print("(lua) JSON Decode Error=", err " at pos=", pos)
    end
    return data
end


function dump(o)
    if type(o) == 'table' then
       local s = '{ '
       for k,v in pairs(o) do
          if type(k) ~= 'number' then k = '"'..k..'"' end
          s = s .. '['..k..'] = ' .. dump(v) .. ','
       end
       return s .. '} '
    else
       return tostring(o)
    end
end


-- This table keeps track of which time corresponds to which fn_call. 
local dotatime_to_step_map = {}
local worldstate_step_offset = nil
local step = 0


function Think()
    step = step + 1
    if GetTeam() == TEAM_DIRE then
        -- For now, just focus on radiant. We can add DIRE action files some time later.
        do return end
    end

    local dotatime = DotaTime()
    local time_ms = dotatime_to_ms(dotatime)
    local gamestate = GetGameState()

    -- Only keep track of this for calibration purposes.
    if live_config == nil then
        dotatime_to_step_map[time_ms] = step
        -- print('dotatime_to_step_map=', dump(dotatime_to_step_map))
    end

    -- print('(lua) Think() dotatime=', dotatime, ' time_ms=', time_ms, 'gamestate=', gamestate, 'step=', step)

    -- To guarantee the dotaservice has received a worldstate, skip this function that amount
    -- of times on first run.
    if step == config.ticks_per_observation then
        -- When we went through exactly this amount calls, it's guaranteed the dotaservice has
        -- received at least one tick, from which we can calibrate.

        local status = {}
        status.dotatime = dotatime
        status.step = step
        print('LUARDY', json.encode(status))

        -- The live configuration gives us back the last time at which dota sent out a 
        -- world state signal.
        live_config = data_from_file(LIVE_CONFIG_FILENAME)
        print('(lua) live_config =', dump(live_config))

        -- We now relate when this was sent out, to the step we were at.
        worldstate_step_offset = dotatime_to_step_map[live_config.calibration_ms]
    end

    if worldstate_step_offset == nil then
        do return end
    end

    if ((step - worldstate_step_offset) % config.ticks_per_observation) == 0 then
        print('(lua) Expecting state @ step=', step, ' @ ms=', time_ms)
        -- TODO(tzaman): read the action file here, and check that it contains an
        -- action with the right timestep.
        local action = get_new_action(time_ms)
        print('(lua) action =', dump(action))
        act(action)
    end
end
