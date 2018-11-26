----------------------------------------------------------------------------------------------------

function GetBotNames ()
	return  {"A", "B", "C", "D", "E"}
end

local bot_heroes = {"npc_dota_hero_nevermore"}

function Think()
	-- This gets called (every server tick AND until all heroes are picked).
	-- This needs to gets called at least once if there is no human.
    print('hero_selection::Think()')
    print('game mode:', GetGameMode())
    local ids = GetTeamPlayers(GetTeam())
    for i,v in pairs(ids) do
        -- If the human is in the unassigned slot, the radiant bots start at v = 2
        -- If the human is in the radiant coach slot, the radiant bots start at v = 2
        -- If the human is in the first radiant slot, the radiant bots start at v = 0
        -- If the human is in the second radiant slot, the radiant bots start at v = 1
        -- If the human is in the third radiant slot, the radiant bots start at v = 2
		if IsPlayerBot(v) and IsPlayerInHeroSelectionControl(v) then
            if i == 1 then
                SelectHero( v, "npc_dota_hero_nevermore" );
            else
                SelectHero( v, "npc_dota_hero_wisp" );
            end
		end
	end
end
