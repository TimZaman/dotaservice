local config = require("bots/config")
local dkjson = require('game/dkjson')

local SELECTION_FILENAME = 'bots/selections_t' .. GetTeam()

local DEFAULT_HEROES = {
    ["pick"] = {
        "npc_dota_hero_nevermore",
        "npc_dota_hero_antimage",
        "npc_dota_hero_lion",
        "npc_dota_hero_bane",
        "npc_dota_hero_rattletrap",
    },
    ["ban"] = {
    },
}

local function get_new_selections(index, phase)
    local file_fn = nil

    phase               = phase or "pick"
    local retIndex      = index
    local retHeroName   = DEFAULT_HEROES[phase][index]

    -- Try to load the file first
    file_fn = loadfile(SELECTION_FILENAME)
    if file_fn ~= nil then
        -- Execute the file_fn; this loads contents into `data`.
        local data = file_fn()
        if data ~= nil then
            local data, pos, err = dkjson.decode(data, 1, nil)
            if err then
                print("(lua) JSON Decode Error=", err, " at pos=", pos)
            else
                local selections = {}
                for indx, selection in pairs(data.actions) do
                    if indx == index then
                        retIndex = indx
                        retHeroName = selection
                    end
                end
            end
        end
    end
    return retIndex, retHeroName
end

local function TeamToChar()
    -- TEAM_RADIANT==2 TEAM_DIRE==3. Makes total sense!
    if GetTeam() == TEAM_RADIANT then return 'R' else return 'D' end
end

function GetBotNames ()
    truncated_team_id = string.sub(config.game_id, 1, 8) .. "_" .. TeamToChar()
	return  {
        truncated_team_id .. "0",
        truncated_team_id .. "1",
        truncated_team_id .. "2",
        truncated_team_id .. "3",
        truncated_team_id .. "4",
    }
end

local function mode1v1()
    local ids = GetTeamPlayers(GetTeam())
    for i,v in pairs(ids) do
        if IsPlayerBot(v) and IsPlayerInHeroSelectionControl(v) and GetSelectedHeroName(v) == "" then
            _, strHero = get_new_selections( i, "pick" )
            SelectHero( v, strHero );
            return
        end
    end
end

local function modeAP()
    local ids = GetTeamPlayers(GetTeam())
end

local function modeCM()
    local ids = GetTeamPlayers(GetTeam())
end

local game_time = 0.0
function Think()
	-- This gets called (every server tick AND until all heroes are picked).
	-- This needs to gets called at least once if there is no human.

    -- NOTE: other remaining game modes of possible interest
    -- GAMEMODE_ARDM -- All Random Death Match
    -- GAMEMODE_ABILITY_DRAFT -- Ability Draft

    local game_mode = GetGameMode()
    -- All Pick
    if game_mode == GAMEMODE_AP then
        modeAP()
    -- Captain's Mode
    elseif game_mode == GAMEMODE_CM then
        modeCM()
    -- 1v1 Mid
    elseif game_mode == GAMEMODE_1V1MID then
        if GameTime() > game_time then
            mode1v1()
            game_time = GameTime() + 5.0
        end
    else
        print("[ERROR] UNKNOWN GAME MODE: ", game_mode)
        do return end
    end
end

-- Function below sets the lane assignments for default bots
-- Obviously, our own agents will do what they belive best
function UpdateLaneAssignments()
    if GetTeam() == TEAM_RADIANT then
        return {
            [1] = LANE_MID,
            [2] = LANE_BOT,
            [3] = LANE_BOT,
            [4] = LANE_TOP,
            [5] = LANE_TOP,
        }
    elseif GetTeam() == TEAM_DIRE then
        return {
            [1] = LANE_MID,
            [2] = LANE_BOT,
            [3] = LANE_BOT,
            [4] = LANE_TOP,
            [5] = LANE_TOP,
        }
    end
end
