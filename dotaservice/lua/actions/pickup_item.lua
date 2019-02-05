-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local PickUpItem= {}

PickUpItem.Name = "Pick Up Item"
PickUpItem.NumArgs = 3

-------------------------------------------------
function PickUpItem:Call( hUnit, iItem, iType )

    iItem = iItem[1]
    iType = iType[1]

    -- TODO: Need to convert itemID to a handle
    -- Below probably will not work unless the
    -- int value given in protobuf maps to the
    -- handle ID (which it does not state it does)
    hItem = iItem
 
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
