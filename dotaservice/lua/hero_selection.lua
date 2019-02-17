local config = require("bots/config")
local pprint = require('bots/pprint')

function TeamToChar()
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

radiant_picks = {}
dire_picks = {}

for _, hero_pick in pairs(config.hero_picks) do
    local hero_id = hero_pick.heroId:lower()
    if hero_pick.teamId == 'TEAM_RADIANT' then
        table.insert(radiant_picks, hero_id)
    else
        table.insert(dire_picks, hero_id)
    end
end


function Think()
	-- This gets called (every server tick AND until all heroes are picked).
	-- This needs to gets called at least once if there is no human.
    local ids = GetTeamPlayers(GetTeam())
    for i, v in pairs(ids) do
        -- If the human is in the unassigned slot, the radiant bots start at v = 2
        -- If the human is in the radiant coach slot, the radiant bots start at v = 2
        -- If the human is in the first radiant slot, the radiant bots start at v = 0
        -- If the human is in the second radiant slot, the radiant bots start at v = 1
        -- If the human is in the third radiant slot, the radiant bots start at v = 2
		if IsPlayerBot(v) and IsPlayerInHeroSelectionControl(v) then
            if GetTeam() == TEAM_RADIANT then
                SelectHero( v, radiant_picks[i]);
            else
                SelectHero( v, dire_picks[i]);
            end
		end
	end
end
