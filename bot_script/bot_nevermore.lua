local dkjson = require( "game/dkjson" )
local config = require("bots/config")

live_config = nil



local function get_live_config()
    -- This function requests a live configuration, other than the static one we had already
    -- imported.
    local ip = '127.0.0.1:'.. tostring(config.port_rest) ..'/calibration'
    local req = CreateRemoteHTTPRequest(ip)
    -- local post_data = '{"foo":5}'
    -- req:SetHTTPRequestRawPostBody("application/json", post_data)
    sent = req:Send(function(result)
        print('-> callback result=', result)
        -- print('-> callback result=', result['Body'])
        -- local data = result['Body']
        local data, pos, err = dkjson.decode(result['Body'], 1, nil)
        if err then
            print("(lua) JSON Decode Error: ", err, " on data: ", result)
        else
            print('Received session settings.')
            live_config = data
            print('live_config=', live_config)
            print('live_config.calibration_tick=', live_config.calibration_tick)
            assert(live_config.calibration_tick < 0)
        end
    end )
end

get_live_config()


local function request_step(tick)
    local ip = '127.0.0.1:'.. tostring(config.port_rest) ..'/step?tick=' .. tostring(tick)
    local req = CreateRemoteHTTPRequest(ip)
    sent = req:Send(function(result)
        -- ?
    end )
end



function dotatime_to_tick(dotatime)
    return math.floor(dotatime * config.ticks_per_second + 0.5)  -- 0.5 for rounding
end


local function get_action_filename(tick)
    return 'bots/' .. config.action_subfolder_name .. '/' .. tostring(tick)
end


local function query_reponse(tick)
    -- Get the response from a file
    local filename = get_action_filename(tick)
    x = 0
    print('(lua) looking for loadfile ', filename)
    while true do
        x = x + 1
        tickfile = loadfile(filename)
        if tickfile ~= nil then break end
    end
    -- print('loadfile retries=', x)
    -- Execute the tickfile; this loads contents into `data`.
    tickfile()
    local data, pos, err = dkjson.decode(data, 1, nil)
    if err then
        print("(lua) JSON Decode Error=", err " at pos=", pos)
    else
        print('(lua) received action:', data)
    end
    return data
end

first_step_requested = false

function Think()
    if GetTeam() == TEAM_RADIANT then
        -- For now, just focus on radiant. We can add DIRE action files some time later.
        return
    end
    local dotatime = DotaTime()
    local tick = dotatime_to_tick(dotatime)
    local gamestate = GetGameState()
    print('(lua) Think() dotatime=', dotatime, ' tick=', tick, 'gamestate=', gamestate)
    -- Notice there is a bug in dotatime, as there is an extra tick inserted at 0;
    -- e.g.: [-0.034439086914062, -0.0011062622070312, 0, 0.033332824707031]

    -- We need to wait until we have the live config for the calibration tick offset.
    if live_config == nil then
        return
    end

    local calibration_tick = live_config.calibration_tick  --  -2671
    
    local tickoffset = calibration_tick % config.ticks_per_observation
    if dotatime > 0 then
        tickoffset = tickoffset -1
    end

    if ((tick - tickoffset) % config.ticks_per_observation) == 0 then
        if first_step_requested == false then
            request_step(tick)
            first_step_requested = true
        end
        query_reponse(tick)
    end

end

