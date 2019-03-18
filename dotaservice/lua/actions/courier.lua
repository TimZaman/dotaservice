-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ActionCourier = {}

ActionCourier.Name = "Courier Action"
ActionCourier.NumArgs = 3

-------------------------------------------------


-- KNOWN COURIER ACTIONS ENUMS
--
-- COURIER_ACTION_RETURN == 0               -- RETURN TO FOUNTAIN
-- COURIER_ACTION_SECRET_SHOP == 1          -- GO TO YOUR SECRET SHOP
-- COURIER_ACTION_RETURN_STASH_ITEMS == 2   -- RETURN ITEMS ON COURIER TO HERO'S STASH
-- COURIER_ACTION_TAKE_STASH_ITEMS == 3     -- TAKE HERO'S STASH ITEMS ONTO COURIER
-- COURIER_ACTION_TRANSFER_ITEMS == 4       -- MOVE ITEMS FROM COURIER TO HERO
-- COURIER_ACTION_BURST == 5                -- PROC INVULN SHIELD
-- COURIER_TAKE_AND_TRANSFER_ITEMS == 6     -- FLY TO HERO AND TRANSFER ITEMS ON COURIER TO HERO
-- COURIER_ACTION_ENEMY_SECRET_SHOP == 7    -- GO TO ENEMY'S SECRET SHOP
-- COURIER_ACTION_SIDESHOP == 8             -- GO TO RADIANT SIDE SHOP
-- COURIER_ACTION_SIDESHOP_2 == 9           -- GO TO DIRE SIDE SHOP


-- NOTE: To move the courier you use a hUnit:MoveToLocation() as per normal unit
--       using the handle to the courier. This function is for Courier specific API

function ActionCourier:Call(hHero, iCourierIndex, iCourierAction)
    -- NOTE: iCourierIndex is 0 indexed (based on C++) and left over
    --       from when you could have multiple courier per team
    local hCourier = GetCourier(iCourierIndex[1] or 0)
    iCourierAction = iCourierAction[1]

    if hCourier and not GetCourierState(hCourier) == COURIER_STATE_DEAD then
        hHero:ActionImmediate_Courier(hCourier, iCourierAction)
    end
end

-------------------------------------------------

return ActionCourier
