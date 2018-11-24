----------------------------------------------------------------------------------------------------

local offset = 0

function Think()
    gs = GetGameState()
    print( "game state: ", gs )
    
    if ( gs == GAME_STATE_HERO_SELECTION ) then
        a = GetGameMode()
        print( "game mode: ", a);
        
        -- if ( a == GAMEMODE_AP ) then 
        --     print ( "All Pick" )
            if ( GetTeam() == TEAM_RADIANT ) then
                    print( "selecting radiant" );
                if ( IsPlayerInHeroSelectionControl(0) ) then
                    SelectHero( 0+offset, "npc_dota_hero_nevermore" );
                end
                
                if ( IsPlayerInHeroSelectionControl(2) ) then
                    SelectHero( 1+offset, "npc_dota_hero_antimage" );
                end
                
                if ( IsPlayerInHeroSelectionControl(3) ) then
                    SelectHero( 2+offset, "npc_dota_hero_antimage" );
                end
                
                if ( IsPlayerInHeroSelectionControl(4) ) then
                    SelectHero( 3+offset, "npc_dota_hero_antimage" );
                end
                
                if ( IsPlayerInHeroSelectionControl(5) ) then
                    SelectHero( 4+offset, "npc_dota_hero_antimage" );
                end
            elseif ( GetTeam() == TEAM_DIRE ) then
                print( "selecting dire" );
                SelectHero( 5+offset, "npc_dota_hero_antimage" );
                SelectHero( 6+offset, "npc_dota_hero_antimage" );
                SelectHero( 7+offset, "npc_dota_hero_antimage" );
                SelectHero( 8+offset, "npc_dota_hero_antimage" );
                SelectHero( 9+offset, "npc_dota_hero_antimage" );
            end
    end
end

----------------------------------------------------------------------------------------------------