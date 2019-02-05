local config = require('bots/config')
local dkjson = require('game/dkjson')
local pprint = require('bots/pprint')
local action_proc = require('bots/action_processor')

local ACTION_FILENAME = 'bots/actions_t' .. GetTeam()
local LIVE_CONFIG_FILENAME = 'bots/live_config_auto'

local debug_text = nil
local live_config = nil


local function act(action)
    local tblActions = {}
    if action.actionType == "DOTA_UNIT_ORDER_NONE" then
        tblActions[action.actionType] = {}
    elseif action.actionType == "DOTA_UNIT_ORDER_MOVE_TO_POSITION" then
        -- NOTE: Move To Position is implemented by Dota2 as a path-navigation movement action.
        --       It will create a list of waypoints that the bot will walk in straight lines between.
        --       The waypoints the system creates will guarantee a valid path between current location
        --       and destination location (PROVIDING A VALID PATH EXISTS).
        --       It approximates reaching each "waypoint" (including last one) before moving to the next 
        --       waypoint (if it exists) with a granularity tested to be 50 units. So Move To Location is
        --       not a VERY precise movement action, but it's not hugely imprecise either. It is an 
        --       important note though, as if you don't check if your position is within the precision
        --       approximiation for movement you could end up instructing the bot to move to the same
        --       location over and over and it just ping-pongs back and forth moving around the precise
        --       location but never directly on it.
        tblActions[action.actionType] = {{action.moveToLocation.location.x, action.moveToLocation.location.y, 0.0}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_MOVE_DIRECTLY" then
        -- NOTE: Move Direclty is implemented by Dota2 as a single point-to-point straight line 
        --       movement action. It does not try to path around any obstacles or check for impossible moves.
        --       It has high precision in final position (gut belief is a 1-2 unit approximation).
        --       Because of how it works, it is ill-advised to use Direct movement for long distances as the
        --       probability of hitting a tree or an obstacle are high and with Direct movement you will
        --       not path around it, but rather get stuck trying to move through it and not succeeding.
        tblActions[action.actionType] = {{action.moveDirectly.location.x, action.moveDirectly.location.y, 0.0}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_ATTACK_TARGET" then
        tblActions[action.actionType] = {{action.attackTarget.target}, {action.attackTarget.once}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_TRAIN_ABILITY" then
        tblActions[action.actionType] = {{action.trainAbility.ability}}
    elseif action.actionType == "DOTA_UNIT_ORDER_GLYPH" then
        tblActions[action.actionType] = {}
    elseif action.actionType == "DOTA_UNIT_ORDER_STOP" then
        tblActions[action.actionType] = {{1}}
    elseif action.actionType == "DOTA_UNIT_ORDER_BUYBACK" then
        tblActions[action.actionType] = {}
    elseif action.actionType == "ACTION_CHAT" then
        tblActions[action.actionType] = {{action.chat.message}, {action.chat.to_allchat}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_POSITION" then
        tblactions[action.actionType] = {{action.castLocation.abilitySlot}, {action.castLocation.location.x, action.castLocation.location.y, 0.0}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_TARGET" then
        tblactions[action.actionType] = {{action.castTarget.abilitySlot}, {action.castTarget.target}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_TARGET_TREE" then
        tblactions[action.actionType] = {{action.castTree.abilitySlot}, {action.castTree.tree}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_NO_TARGET" then
        tblactions[action.actionType] = {{action.cast.abilitySlot}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_TOGGLE" then
        tblactions[action.actionType] = {{action.castToggle.abilitySlot}}
    elseif action.actionType == "DOTA_UNIT_ORDER_PICKUP_RUNE" then
        tblactions[action.actionType] = {{action.pickUpRune.rune}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_PICKUP_ITEM" then
        tblactions[action.actionType] = {{action.pickUpItem.itemId}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_DROP_ITEM" then
        tblactions[action.actionType] = {{action.dropItem.slot}, {action.dropItem.location.x, action.dropItem.location.y, 0,0}, {0}}
    end
    action_proc:Run(GetBot(), tblActions)
end


local function get_new_actions(dota_time, player_id, curr_step)
    -- print('(lua) get_new_action @', dota_time, ' player_id=', player_id)
    -- print('ACTION_FILENAME=', ACTION_FILENAME)
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
            if data == 'FLUSH' then
                return nil
            end
            local data, pos, err = dkjson.decode(data, 1, nil)
            if err then
                print("(lua) JSON Decode Error=", err, " at pos=", pos)
            end
            
            if data.dotaTime == dota_time or data.dotaTime == nil then
                -- Now find correponding player id
                local actions = {}
                for _, action in pairs(data.actions) do
                    if action.player == player_id then
                        if action.actionDelay == nil then
                            -- Make sure this defaults to 0 if not present
                            action.actionDelay = curr_step
                        else
                            action.actionDelay = curr_step + action.actionDelay
                        end
                        table.insert(actions, action)
                    end
                end
                return actions
            end
        end
    end
end


local function data_from_file(filename)
    -- Get the response from a file, while waiting for the file.
    print('(lua) looking for loadfile =', filename)
    local file_fn = nil
    while true do
        while true do
            file_fn = loadfile(filename)
            if file_fn ~= nil then break end
        end
        -- Execute the file_fn; this loads contents into `data`.
        local data = file_fn()
        if data ~= nil then
            local data, pos, err = dkjson.decode(data, 1, nil)
            if err then
                print("(lua) JSON Decode Error=", err " at pos=", pos)
            end
            return data
        end
    end
end


-- This table keeps track of which time corresponds to which fn_call. 
local dota_time_to_step_map = {}
local worldstate_step_offset = nil
local step = 0

local actions = {}

function Think()
    
    step = step + 1

    -- if GetTeam() == TEAM_DIRE then
    --     -- For now, just focus on radiant. We can add DIRE action files some time later.
    --     do return end
    -- end
    
    -- TODO (tzaman): Get controllable bot IDs here somehow
    if GetBot():GetPlayerID() ~= 0 and GetBot():GetPlayerID() ~= 5  then
        do return end
    end

    local dota_time = DotaTime()
    local game_state = GetGameState()

    -- Only keep track of this for cliabration purposes.
    if live_config == nil then
        dota_time_to_step_map[dota_time] = step
    end
    -- print('bot id=', GetBot():GetPlayerID(), 'step=', step, ' time=', DotaTime())

    -- x = 10000.4
    -- for i=1,100000000 do x = x/2.2 end
    -- print('(lua) Think() dota_time=', dota_time, 'game_state=', game_state, 'step=', step)

    -- To guarantee the dotaservice has received a worldstate, skip this function that amount
    -- of times on first run. Take twice the amount to be safe.
    if step == config.ticks_per_observation * 2 then
        -- When we went through exactly this amount calls, it's guaranteed the dotaservice has
        -- received at least one tick, from which we can calibrate.

        local status = {}
        status.dota_time = dota_time
        status.step = step
        print('LUARDY', json.encode(status))

        -- The live configuration gives us back the last time at which dota sent out a 
        -- world state signal.
        live_config = data_from_file(LIVE_CONFIG_FILENAME)
        print('(lua) received live_config =', pprint.pformat(live_config))

        -- We now relate when this was sent out, to the step we were at.
        worldstate_step_offset = dota_time_to_step_map[live_config.calibration_dota_time]
        -- print('worldstate_step_offset:', worldstate_step_offset)
    end

    if worldstate_step_offset == nil then
        do return end
    end

    if ((step - worldstate_step_offset) % config.ticks_per_observation) == 0 then
        -- print('(lua) Expecting state @ step=', step, 'dota_time=', dota_time)
        -- TODO(tzaman): read the action file here, and check that it contains an
        -- action with the right timestep.
        actions = get_new_actions(dota_time, GetBot():GetPlayerID(), step)
        -- print('(lua) received actions =', dump(actions))

        debug_text = pprint.pformat(actions)
    end

    if actions and #actions > 0 then
        completed_actions = {}
        for index, action in pairs(actions) do
            if step == action.actionDelay then
                act(action)
                table.insert(completed_actions, 1, inded)
            end
        end

        for _, erase_id in pairs(completed_actions) do
            table.remove(actions, erase_id)
        end
    end

    if debug_text ~= nil then
        DebugDrawText(8, 90, debug_text, 255, 255, 255)
    end
end
