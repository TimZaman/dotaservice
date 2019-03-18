-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local PickUpItem= {}

PickUpItem.Name = "Pick Up Item"
PickUpItem.NumArgs = 4

-------------------------------------------------
function PickUpItem:Call( hUnit, iItem, vLoc, iType )

    iItem = iItem[1]
    iType = iType[1]
    vLoc = Vector(vLoc[1], vLoc[2], vLoc[3])

    -- Get list of all dropped items
    droppedItemTable = GetDroppedItemList()
    hItem = nil
    -- Iterate list and match on location
    for indx, hTable in pairs(droppedItemTable) do
        -- hTable is { hItem, hOwner, iPlayer, vLoc }
        if hTable and hTable[4] then
            if hTable[4].x == vLoc.x and hTable[4].y == vLoc.y then
                hItem = hTable[1]
            end
        end
    end
 
    if not hItem then
        do return end
    end

    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_PickUpItem(hItem)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_PickUpItem(hItem)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_PickUpItem(hItem)
    end
end
-------------------------------------------------

return PickUpItem
