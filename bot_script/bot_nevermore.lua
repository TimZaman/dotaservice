-- Add the ramdisk to the path
package.path = package.path .. ";/Volumes/ramdisk/?.lua"

dkjson = require( "game/dkjson" )

print(package.path)
print('LUA version:', _VERSION)

-- f = loadfile("../foos")
-- f = loadfile("ramdisk/foos")
-- f = loadfile("/Volumes/ramdisk/foos")
-- f();
-- require('foos')
-- exit();


function OnStart()
	print( "bot_nevermore.OnStart" );
end

----------------------------------------------------------------------------------------------------

function OnEnd()
	print( "bot_nevermore.OnEnd" );
end

----------------------------------------------------------------------------------------------------

web_config = {}
web_config.IP_ADDR = "127.0.0.1"
web_config.IP_BASE_PORT = 5000

game_id = 'my_game_id'

local ip_addr = web_config.IP_ADDR .. ":" .. web_config.IP_BASE_PORT

N_READ_RETRIES = 10000
TICKS_PER_SECOND = 30
PACKET_EVERY_N_TICKS = 4
ACTION_FOLDER = 'ramdisk/'

function dotatime_to_tick(dotatime)
    return math.floor(dotatime * TICKS_PER_SECOND + 0.5)  -- 0.5 for rounding
    -- return math.floor(math.round(dotatime * TICKS_PER_SECOND), 
end

-- local function http_request(tick, post_data)
--     local ip_addr = web_config.IP_ADDR .. ":" .. tostring(web_config.IP_BASE_PORT) .. "/action?game_id=" .. game_id .. "&tick=" .. tick
--     local req = CreateRemoteHTTPRequest( ip_addr )
--     req:SetHTTPRequestRawPostBody("application/json", post_data)
--     sent = req:Send(function(result)
--         -- TODO(tzaman): Check result for error
--     end )
--     -- print('sent:', sent)
-- end

local function get_action_filename(tick)
    return ACTION_FOLDER .. game_id .. '/' .. tostring(tick)
end


local function query_reponse(tick)
    -- Perform a HTTP request to the server with the next action for given tick
    local post_data = '{}'
    -- local reponse = http_request(tick, post_data)
    -- Get the response from a file
    local filename = get_action_filename(tick)
    -- print('looking for filename=', filename)
    x = 0
    --for i = 1, N_READ_RETRIES, 1 do -- Dota roundtrip takes around 70 tries.
    print('looking for loadfile ', filename)
    while true do
        x = x + 1
        tickfile = loadfile(filename)
        if tickfile ~= nil then break end
    end
    print('loadfile retries=', x)
    if tickfile == nil then
        print('tickfile is nil; timed out waiting for the file, or invalid contents')
        print('skipping tickfile.')
        data = nil
    else
        -- Execute the tickfile; this loads contents into `data`.
        tickfile()
    end
    print('data=', data)
    local jsonReply, pos, err = dkjson.decode(data, 1, nil)
    if err then
        print("JSON Decode Error: ", err)
        print("Sent Message: ", json)
        print("Msg Body: ", v)
    else
        print('Decode sucessful:', jsonReply)
    end
    return data
end

function Think()
    local dotatime = DotaTime()
    local tick = dotatime_to_tick(dotatime)
    print('Think dotatime=', dotatime, ' tick=', tick)
    -- Notice there is a bug in dotatime, as there is an extra tick inserted at 0;
    -- e.g.: [-0.034439086914062, -0.0011062622070312, 0, 0.033332824707031]
    if dotatime >= 0 and tick % 4 == 0 then
        query_reponse(tick)
    end
end

