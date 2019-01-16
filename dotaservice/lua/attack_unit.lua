-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local AttackUnit = {}

AttackUnit.Name = "Attack Unit"
AttackUnit.NumArgs = 4

local function toboolean(number)
    if number >= 1 then return true end
    return false
end

-------------------------------------------------
function AttackUnit:Call( hUnit, hTarget, bOnce, iType )
    hTarget = hTarget[1]
    if hTarget == -1 then -- Invalid target. Do nothing.
        do return end
    end
    hTarget = GetBotByHandle(hTarget)

    iType = iType[1]
    
    if not hTarget:IsNull() then
        vLoc = hTarget:GetLocation()
        DebugDrawCircle(vLoc, 25, 255, 0, 0)
        DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)
        
        bOnce = toboolean(bOnce[1])
        if iType == nil or iType == ABILITY_STANDARD then
            hUnit:Action_AttackUnit(hTarget, bOnce)
        elseif iType == ABILITY_PUSH then
            hUnit:ActionPush_AttackUnit(hTarget, bOnce)
        elseif iType == ABILITY_QUEUE then
            hUnit:ActionQueue_AttackUnit(hTarget, bOnce)
        end
    end
end
-------------------------------------------------

return AttackUnit
