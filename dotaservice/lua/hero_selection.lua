local config = require("bots/config")

function TeamToChar()
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
        -- If the human is in the unassigned slot, the radiant bots start at v = 2
        -- If the human is in the radiant coach slot, the radiant bots start at v = 2
        -- If the human is in the first radiant slot, the radiant bots start at v = 0
        -- If the human is in the second radiant slot, the radiant bots start at v = 1
        -- If the human is in the third radiant slot, the radiant bots start at v = 2
        if IsPlayerBot(v) and IsPlayerInHeroSelectionControl(v) and GetSelectedHeroName(v) == "" then
        -- if i == 1 and GetTeam() == TEAM_RADIANT then
            if i == 1 then
                SelectHero( v, "npc_dota_hero_sniper" );
            else
                SelectHero( v, "npc_dota_hero_zuus" );
            end
        end
    end
end

local function modeAP()
    local ids = GetTeamPlayers(GetTeam())
end

local function modeCM()
    local ids = GetTeamPlayers(GetTeam())
end

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
        mode1v1()
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
