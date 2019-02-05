-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local PickUpRune= {}

PickUpRune.Name = "Pick Up Rune"
PickUpRune.NumArgs = 3

-------------------------------------------------
function PickUpRune:Call( hUnit, iRune, iType )

    iRune = iRune[1]
    iType = iType[1]
 
    vLoc = GetRuneSpawnLocation(iRune)
    DebugDrawCircle(vLoc, 25, 255, 0, 0)
    DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)
    
    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_PickUpRune(iRune)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_PickUpRune(iRune)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_PickUpRune(iRune)
    end
end
-------------------------------------------------

return PickUpRune
