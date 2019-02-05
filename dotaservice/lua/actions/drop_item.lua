-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local DropItem= {}

DropItem.Name = "Drop Item"
DropItem.NumArgs = 4

-------------------------------------------------
function DropItem:Call( hUnit, iSlot, vLoc, iType )
    iSlot = iSlot[1]
    hItem = GetItemInSlot(iSlot)
    if hItem == nil then
        print("[Action::DropItem] Error - invalid handle for slot: ", iSlot)
        do return end
    end

    vLoc = Vector(vLoc[1], vLoc[2], vLoc[3])
    iType = iType[1]

    DebugDrawCircle(vLoc, 25, 255, 255 ,255)
    DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 255, 255)

    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_DropItem(hItem, vLoc)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_DropItem(hItem, vLoc)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_DropItem(hItem, vLoc)
    end
end
-------------------------------------------------

return DropItem
