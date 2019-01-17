-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local UseAbility = {}

UseAbility.Name = "Use Ability"
UseAbility.NumArgs = 3

-------------------------------------------------
function UseAbility:Call( hUnit, intAbilitySlot, iType )
    local hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])
    if not hAbility then
        print('[ERROR]: ', hUnit:GetUnitName(), " failed to find ability in slot ", intAbilitySlot[1])
        do return end
    end

    iType = iType[1]

    -- Note: we do not test for range, mana/cooldowns or any debuffs on the hUnit (e.g., silenced).

    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_UseAbility(hAbility)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_UseAbility(hAbility)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_UseAbility(hAbility)
    end
end
-------------------------------------------------

return UseAbility
