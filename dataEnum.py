from enum import Enum

tileCategories = [
    'ore',
    'gem',
    'small_plants',
    'large_plants',
    'banner',
    'loot_pot',
    'crafting_station',
    'relic',
    'tree',
    'fishing_crate',
    'platform',  # And planter boxes
    'pylon',
    'statue',
    'lighting',    # Torch, campfire
    'boss_summon',  # Plantera bulb, crimson heart, etc.
    'painting',
    'buff_station',  # Cake, ammo box
    'mechanism',   # Any item that uses wiring
    'boulder',
    'rubble',   # Decorative tiles that don't drop anything
    'other',
    # The below is unused
    'common',   # Dirt, stone
    'biome',    # Biome specific: mud, hive block
    'furniture',
]

class TileID(Enum):
    DIRT = 0
    STONE = 1
    GRASS = 2
    PLANTS_FOREST = 3
    TREE = 5
    IRON_ORE = 6
    COPPER_ORE = 7
    GOLD_ORE = 8
    SILVER_ORE = 9
    DOOR_CLOSED = 10
    DOOR_OPEN = 11
    LIFE_CRYSTAL = 12
    BOTTLE = 13
    TABLE = 14
    CHAIR = 15
    ANVIL = 16
    FURNACE = 17
    WORKBENCH = 18
    PLATFORM = 19
    SAPLING = 20
    CHEST = 21
    DEMONITE_ORE = 22
    GRASS_CORRUPT = 23
    PLANTS_CORRUPT = 24
    EBONSTONE_BLOCK = 25
    ALTAR = 26
    SUNFLOWER = 27
    POT = 28
    PIGGY_BANK = 29
    WOOD = 30
    EVIL_ORB = 31
    THORN_CORRUPT = 32
    CANDLE = 33
    CHANDELIER = 34
    JACK_O_LANTERN = 35
    PRESENT = 36
    METEORITE = 37
    GRAY_BRICK = 38
    RED_BRICK = 39
    CLAY = 40
    BLUE_BRICK = 41
    LANTERN = 42
    GREEN_BRICK = 43
    PINK_BRICK = 44
    GOLD_BRICK = 45
    SILVER_BRICK = 46
    COPPER_BRICK = 47
    SPIKE = 48
    WATER_CANDLE = 49
    BOOK = 50
    COBWEB = 51
    VINE = 52
    SAND = 53
    GLASS = 54
    SIGN = 55
    OBSIDIAN = 56
    ASH = 57
    HELLSTONE = 58
    MUD = 59
    GRASS_JUNGLE = 60
    PLANTS_JUNGLE = 61
    VINE_JUNGLE = 62
    SAPPHIRE_STONE_BLOCK = 63
    RUBY_STONE_BLOCK = 64
    EMERALD_STONE_BLOCK = 65
    TOPAZ_STONE_BLOCK = 66
    AMETHYST_STONE_BLOCK = 67
    DIAMOND_STONE_BLOCK = 68
    THORN_JUNGLE = 69
    GRASS_MUSHROOM = 70
    PLANTS_MUSHROOM = 71
    TREE_MUSHROOM = 72
