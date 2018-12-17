-- By putting this file and implementing the
-- Think() function in bot_generic.lua we 
-- effectively prevent the Think() function
-- of all bots from happening as coded by
-- Valve. Bot behavior can still be controlled
-- by creating bot_<heroName>.lua files.

function Think()
    return
end
