-- This file prevents default Valve bot behavior
-- for item & ability usage. In addition, by 
-- including courier, buyback and AbilityLevelUp
-- Think() we prevent those from happenging
-- as well.
--

function ItemUsageThink()
    return
end

function AbilityUsageThink()
    return
end

function CourierUsageThink()
    return
end

function BuybackUsageThink()
    return
end

function AbilityLevelUpThink()
    return
end
