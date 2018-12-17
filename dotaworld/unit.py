
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

def isHeroUnit(unit):
    return unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO')

def isLaneCreepUnit(unit):
    return unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP')

def isJungleUnit(unit):
    return unit.unit_type == CMsgBotWorldState.UnitType.Value('JUNGLE_CREEP')

def isCourier(unit):
    return unit.unit_type == CMsgBotWorldState.UnitType.Value('COURIER')

def isWard(unit):
    return unit.unit_type == CMsgBotWorldState.UnitType.Value('WARD')

def isRoshan(unit):
    return unit.unit_type == CMsgBotWorldState.UnitType.Value('ROSHAN')


# Inferred Units
def isCreepUnit(unit):
    return isLaneCreepUnit(unit) or isJungleUnit(unit)
