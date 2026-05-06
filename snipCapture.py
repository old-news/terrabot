from PIL import Image
import json
from math import floor, ceil
import numpy as np
import os
import cv2
from functools import lru_cache
# from pathlib import Path
import random
# import time
import math


trainingPath = './training'
validationPath = './validation'
BLOCK_MIN_BRIGHTNESS = 0.1


@lru_cache(maxsize=3000)
def getImage(path):
    return Image.open(path)


@lru_cache(maxsize=3000)
def getImageArray(path):
    return np.array(getImage(path))


def areNpArraysSimilar(arr1, arr2):
    diffM = np.abs(arr1 + arr2)
    diff = np.mean(diffM)
    return diff < 2


def getIDcategory(numID: int) -> str:
    if numID in [6, 7, 8, 9, 22, 37, 56, 58, 107, 108, 111, 166, 167, 168, 169, 204, 211, 221, 222, 223, 408]:
        return 'ore'
    elif numID in [178]:
        return 'gem'
    elif numID in [3, 24, 61, 71, 73, 74, 110, 113, 201, 637, 703]:
        return 'small_plants'
    elif numID in [227, 233, 530, 651, 652, 705]:
        return 'large_plants'
    elif numID in [91, 465]:
        return 'banner'
    elif numID in [28, 653]:
        return 'loot_pot'
    elif numID in [16, 17, 18, 13, 26, 77, 86, 94, 96, 101, 106, 114, 125, 133, 134, 172, 207, 217, 218, 219, 220, 228, 237, 243, 247, 283, 300, 301, 302, 303, 304, 305, 306, 307, 308, 355, 412, 499, 622, 642, 695, 699]:
        return 'crafting_station'
    elif numID in [617]:
        return 'relic'
    elif numID in [5, 20, 72, 323, 571, 583, 584, 585, 586, 587, 588, 589, 590, 595, 596, 615, 616, 634]:
        # And saplings
        return 'tree'
    elif numID in [376]:
        return 'fishing_crate'
    elif numID in [19, 380, 427, 435, 436, 437, 438, 439]:
        # And planter boxes (id=380)
        return 'platform'
    elif numID in [597]:
        return 'pylon'
    elif numID in [105, 337, 349, 531]:
        return 'statue'
    elif numID in [33, 49, 100, 174, 372, 646] or numID in [100, 173] or numID in [42, 92, 93, 100, 390, 564, 34, 95, 98] or numID in [4, 93] or numID in [35, 126, 149, 215, 405, 592]:
        return 'lighting'
    elif numID in [31, 129, 231, 237, 238]:
        # id 129 gelatin crystal and crystal shards
        return 'boss_summon'
    elif numID in [240, 241, 242, 245, 246]:
        return 'painting'
    elif numID in [287, 354, 377, 464, 621]:
        return 'buff_station'
    elif numID in [131, 132, 135, 136, 137, 141, 142, 143, 144, 210, 216, 235, 314, 335, 338, 411, 419, 420, 421, 422, 423, 424, 425, 428, 429, 442, 443, 445]:
        return 'mechanism'
    elif numID in [138, 664, 665, 711, 712, 713, 714, 715, 716]:
        return 'boulder'
    elif numID in [165, 185, 186, 187, 647, 648, 649, 650, 693, 694, 704, 705]:
        return 'rubble'
    else:
        return 'other'


def classifyTile(tileInfo: dict):
    tileID = tileInfo['type']
    if tileID in [441, 467]:
        # Variants of chest (dead man's, trapped)
        tileID = 21
    fX = tileInfo['fX']
    fY = tileInfo['fY']
    relevantExtractor = {
        (3, 144, 0): 'mushroom',
        (110, 144, 0): 'mushroom',  # In hallowed biome
        (201, 270, 0): 'vicious_mushroom',
        (24, 144, 0): 'vile_mushroom',
        (5, 0, 0): 'tree_base',
        (5, 0, 22): 'tree_base',
        (5, 0, 44): 'tree_base',
        (5, 0, 132): 'tree_base',
        (5, 0, 154): 'tree_base',
        (5, 0, 176): 'tree_base',
        (5, 66, 132): 'tree_base',
        (5, 66, 154): 'tree_base',
        (5, 66, 176): 'tree_base',
        (5, 88, 132): 'tree_base',
        (5, 88, 154): 'tree_base',
        (5, 88, 176): 'tree_base',
        (596, 0, 88): 'sakura_base',
        (596, 66, 132): 'sakura_base',
        (596, 88, 176): 'sakura_base',
        (616, 0, 22): 'yellow_willow_base',
        (634, 0, 0): 'ashtree_base',
        (634, 0, 22): 'ashtree_base',
        (634, 0, 44): 'ashtree_base',
        (634, 0, 132): 'ashtree_base',
        (634, 0, 154): 'ashtree_base',
        (634, 0, 176): 'ashtree_base',
        (634, 66, 132): 'ashtree_base',
        (634, 66, 154): 'ashtree_base',
        (634, 66, 176): 'ashtree_base',
        (634, 88, 132): 'ashtree_base',
        (634, 88, 154): 'ashtree_base',
        (634, 88, 176): 'ashtree_base',
        (323, 66, 0): 'palm_base',
        (80, 0, 36): 'cactus_base',
        (13, 0, 0): 'placed_bottle',
        (13, 18, 0): 'placed_lesser_healing_potion',
        (13, 36, 0): 'placed_lesser_mana_potion',
        (21, 36, 0): 'gold_chest',
        (21, 36, 18): 'gold_chest',
        (21, 54, 0): 'gold_chest',
        (21, 54, 18): 'gold_chest',
        (21, 72, 0): 'locked_gold_chest',
        (21, 72, 18): 'locked_gold_chest',
        (21, 90, 0): 'locked_gold_chest',
        (21, 90, 18): 'locked_gold_chest',
        (21, 108, 0): 'shadow_chest',
        (21, 108, 18): 'shadow_chest',
        (21, 126, 0): 'shadow_chest',
        (21, 126, 18): 'shadow_chest',
        (21, 144, 0): 'locked_shadow_chest',
        (21, 144, 18): 'locked_shadow_chest',
        (21, 162, 0): 'locked_shadow_chest',
        (21, 162, 18): 'locked_shadow_chest',
        (50, 90, 0): 'placed_water_bolt',
        (185, 576, 18): 'rubble_copper_bag',
        (50, 90, 0): 'placed_water_bolt',
        (185, 576, 18): 'rubble_copper_bag',
        (50, 90, 0): 'placed_water_bolt',
        (185, 576, 18): 'rubble_copper_bag',
        (50, 90, 0): 'placed_water_bolt',
        (185, 576, 18): 'rubble_copper_bag',
        (185, 594, 18): 'rubble_copper_bag',
        (61, 162, 0): 'natures_gift',
        (61, 144, 0): 'jungle_spore',
        (583, 0, 22): 'gemtree_topaz_base',
        (584, 0, 22): 'gemtree_amethyst_base',
        (585, 0, 22): 'gemtree_sapphire_base',
        (586, 0, 22): 'gemtree_emerald_base',
        (587, 0, 22): 'gemtree_ruby_base',
        (588, 0, 22): 'gemtree_diamond_base',
        (589, 0, 22): 'gemtree_amber_base',
        (590, 36, 0): 'topaz_sapling',
        (590, 36, 18): 'topaz_sapling',
        (590, 90, 0): 'amethyst_sapling',
        (590, 90, 18): 'amethyst_sapling',
        (590, 108, 0): 'sapphire_sapling',
        (590, 108, 18): 'sapphire_sapling',
        (590, 180, 0): 'emerald_sapling',
        (590, 180, 18): 'emerald_sapling',
        (590, 252, 0): 'ruby_sapling',
        (590, 252, 18): 'ruby_sapling',
        (590, 306, 0): 'diamond_sapling',
        (590, 306, 18): 'diamond_sapling',
        (590, 324, 0): 'amber_sapling',
        (590, 324, 18): 'amber_sapling',
    }
    categorized = relevantExtractor.get((tileID, fX, fY))
    if categorized is not None:
        categorized = f'{tileID}_{categorized}'
    else:
        # 91 is all banners
        # Not all of them are mob banners, soooo it would make sense to classify all of them
        # however, there are so many that i don't really want to
        # Sam with statues (id=105) and gems(id=178)
        # Mechanisms = 135, 137, 144, 235, 419, 420, 421, 422, 423, 424, 428, 429, 443
        # 209=cannons
        # 376=fishing crates
        if tileID in [4, 82, 83, 84, 91, 105, 135, 137, 144, 178, 209, 235, 239, 376, 419, 420, 421, 422, 423, 424, 428, 429, 443]:
            categorized = f'{tileID}/{fX}_{fY}'
        elif 254 == tileID:
            # Pumpkin
            if fX >= 108:
                categorized = 'mature_pumpkin'
            else:
                categorized = 'seedling_pumpkin'
        elif 597 == tileID:
            # Is pylon
            if 0 <= fX <= 36:
                categorized = 'forest_pylon'
            elif 54 <= fX <= 90:
                categorized = 'jungle_pylon'
            elif 108 <= fX <= 144:
                categorized = 'hallowed_pylon'
            elif 162 <= fX <= 198:
                categorized = 'cavern_pylon'
            elif 216 <= fX <= 252:
                categorized = 'ocean_pylon'
            elif 270 <= fX <= 306:
                categorized = 'desert_pylon'
            elif 324 <= fX <= 360:
                categorized = 'snow_pylon'
            elif 378 <= fX <= 414:
                categorized = 'mushroom_pylon'
            elif 432 <= fX <= 468:
                categorized = 'universal_pylon'
        elif 215 == tileID:
            # Is campfire
            if 0 <= fX <= 36:
                categorized = 'original_campfire'
            elif 54 <= fX <= 90:
                categorized = 'cursed_campfire'
            elif 108 <= fX <= 144:
                categorized = 'demon_campfire'
            elif 162 <= fX <= 198:
                categorized = 'frozen_campfire'
            elif 216 <= fX <= 252:
                categorized = 'ichor_campfire'
            elif 270 <= fX <= 306:
                categorized = 'rainbow_campfire'
            elif 324 <= fX <= 360:
                categorized = 'ultrabright_campfire'
            elif 378 <= fX <= 414:
                categorized = 'bone_campfire'
            elif 432 <= fX <= 468:
                categorized = 'desert_campfire'
            elif 486 <= fX <= 522:
                categorized = 'coral_campfire'
            elif 594 <= fX <= 630:
                categorized = 'crimson_campfire'
            elif 648 <= fX <= 684:
                categorized = 'hallowed_campfire'
            elif 702 <= fX <= 738:
                categorized = 'jungle_campfire'
            elif 756 <= fX <= 792:
                categorized = 'mushroom_campfire'
            elif 810 <= fX <= 846:
                categorized = 'aether_campfire'
    if categorized is None: categorized = tileID
    return categorized


def classifyBlock(tileInfo: dict) -> str:
    # Return in this order: unknown, tile, liquid, wall, air
    lighting = tileInfo['lighting']
    brightness = sum(lighting) / 3 / 255
    isDark = brightness < BLOCK_MIN_BRIGHTNESS  # Block is too dark for a human to distinguish
    tileID = tileInfo['type']
    if tileInfo['hasTile'] is not False:
        if isDark: return 'tile/unknown'
        category = getIDcategory(tileID)
        tileClass = classifyTile(tileInfo)
        return f'tile/{category}/{tileClass}'
    if tileInfo['liquidAmount'] / 255 >= 0.2 and (tileInfo['isSolid'] is True or tileInfo['isActuated'] is True):
        # Liquid
        if isDark: return 'liquid/unknown'
        liquidType = tileInfo['liquidType']
        # liquidAmount = tileInfo['liquidAmount']
        return f'liquid/{liquidType}'
    wallID = tileInfo['wallType']
    if wallID != 0:
        if isDark: return 'wall/unknown'
        return f'wall/{wallID}'
    if isDark: return 'air/unknown'
    return 'air/0'


def saveClassifyMiddleSnip(img: np.ndarray, captureData: dict) -> None:
    #img[284:484, 583:783]
    middleImg = img[284:484, 583:783]
    x_adjust = 16 - int(captureData['ScreenPosX']) % 16
    y_adjust = 16 - int(captureData['ScreenPosY']) % 16
    xPath = f'{trainingPath if random.random() > 0.2 else validationPath}/offset_x/{x_adjust}/'
    yPath = f'{trainingPath if random.random() > 0.2 else validationPath}/offset_y/{y_adjust}/'
    if not os.path.exists(xPath):
        os.makedirs(xPath)
    if not os.path.exists(yPath):
        os.makedirs(yPath)
    cv2.imwrite(f'{xPath}/{len(os.listdir(xPath))}.png', middleImg)
    cv2.imwrite(f'{yPath}/{len(os.listdir(yPath))}.png', middleImg)


def tileImage(img: np.ndarray, captureData) -> np.ndarray:
    height, width, channels = img.shape
    tileWidth, tileHeight = floor(width / 16) - 1, floor(height / 16) - 1
    tileImages = [[0 for x in range(tileWidth - 2)] for y in range(tileHeight - 2)]
    x_adjust = 16 - int(captureData['ScreenPosX']) % 16
    y_adjust = 16 - int(captureData['ScreenPosY']) % 16
    #x_adjust = 0
    #y_adjust = 0
    #x_adjust = -16 * int(x_adjust < 8)
    #y_adjust = 0  # -16 * int(y_adjust < 8)
    for x in range(1, tileWidth - 2):
        for y in range(1, tileHeight - 2):
            #if isInNogoZone: continue
            #newtilePixelArray = img[16*y + y_adjust:16*(y+1) + y_adjust, 16*x + x_adjust:16*(x+1) + x_adjust, :]
            newtilePixelArray = img[16*y+y_adjust:16*(y+1)+y_adjust, 16*x+x_adjust:16*(x+1)+x_adjust, :]
            tileImages[y][x] = newtilePixelArray
    return tileImages


def parseTileData(tileData):
    out = []
    for col in tileData:
        out.append([])
        for tile in col:
            tileInfo = {
                'type': tile[0],
                'fX': tile[1],
                'fY': tile[2],
                'lighting': [tile[3], tile[4], tile[5]],
                'liquidAmount': tile[6],
                'liquidType': tile[7],
                'wallType': tile[8],
                'wallfX': tile[9],
                'wallfY': tile[10],
                'hasTile': tile[11] == 1,
                'isActuated': tile[12] == 1,
                'isSolid': tile[13] == 1
            }
            out[-1].append(tileInfo)
    return out


def saveTiles(tileImages: np.ndarray, captureData: dict, nogoZone=None) -> None:
    classifiedImageCounts = dict()
    tileData = parseTileData(captureData['TileData'])
    # classifiedImageCounts is used because os.listdir takes a lot of time
    for mainPath in os.listdir(trainingPath):
        for category in os.listdir(f'{trainingPath}/{mainPath}'):
            ids = os.listdir(f'{trainingPath}/{mainPath}/{category}')
            if len(ids) == 0:
                classifiedImageCounts[f'{mainPath}/{category}'] = 0
                continue
            if os.path.isdir(f'{trainingPath}/{mainPath}/{category}/{ids[0]}'):
                for ID in ids:
                    classifiedImageCounts[f'{mainPath}/{category}/{ID}'] = len(os.listdir(f'{trainingPath}/{mainPath}/{category}/{ID}'))
            else:
                classifiedImageCounts[f'{mainPath}/{category}'] = len(os.listdir(f'{trainingPath}/{mainPath}/{category}'))
    xmod = captureData['ScreenPosX'] % 16
    ymod = captureData['ScreenPosY'] % 16
    x_adjust = 1 + int(xmod >= 8)
    y_adjust = 1 + int(ymod >= 8)
    for y, imageRow in enumerate(tileImages):
        for x, image in enumerate(imageRow):
            isInNogoZone = False
            if nogoZone is not None:
                corners = ((16*x, 16*y), (16*(x+3), 16*y), (16*x, 16*(y+3)), (16*(x+3), 16*(y+3)))
                for zone in nogoZone:
                    zcorners = ((zone[0][0], zone[0][1]), (zone[0][0], zone[1][1]), (zone[1][0], zone[0][1]), (zone[1][0], zone[1][1]))
                    if isInNogoZone: break
                    for corner in corners:
                        if corner[0] > zone[0][0] and corner[0] < zone[1][0] and corner[1] > zone[0][1] and corner[1] < zone[1][1]:
                            isInNogoZone = True
                            break
                    if isInNogoZone: break
                    for zcorner in zcorners:
                        if zcorner[0] > corners[0][0] and zcorner[0] < corners[3][0] and zcorner[1] > corners[0][1] and zcorner[1] < corners[3][1]:
                            isInNogoZone = True
                            break
            if isInNogoZone: continue
            #print(imageRow)
            #if not isinstance(image, np.ndarray): continue
            # if np.mean(image) < 25: continue    # Image is too dark to tell what it is
            # ABOVE IS DEPRECATED: using tile lighting from tModLoader to determine if tile is too dark
            tileInfo = tileData[x][y]
            classified = classifyBlock(tileInfo)
            mainPlace = trainingPath if random.random() > 0.2 else validationPath
            path = f'{mainPlace}/{classified}'
            imgAlreadyInSet = False
            nameID = classifiedImageCounts.get(classified)
            if nameID is None:
                nameID = 0
            if not os.path.exists(path):
                os.makedirs(path)
                mainPlace = trainingPath
                path = f'{mainPlace}/{classified}'
            if imgAlreadyInSet: continue
            if random.random() > 1 / (math.e ** ((nameID - 500) / 500)):
                # Do not store a crap ton of images for common dataIDs (like dirt)
                continue
            classifiedImageCounts[classified] = nameID + 1
            cv2.imwrite(f'{path}/{nameID}-{xmod}_{ymod}.png', image)


def snipImageAndSaveClassified(img: np.ndarray, captureData, endMessage=None, nogoZone=None) -> None:
    saveClassifyMiddleSnip(img, captureData)
    if captureData['IsInventoryOpen']: return
    if nogoZone is None:
        # This nogoZone is formatted for a UI scale of 78%
        cursorPos = captureData['CursorPos']
        nogoZone = [
            ((1135, 13), (1330, 63)),   # Health
            ((1335, 16), (1360, 197)),  # Mana
            ((17, 0), (357, 56)),       # Hotbar
            ((25, 60), (348, 134)),     # Two rows of buffs. Four rows goes to y=211
            ((660, 405), (707, 465)),    # The player
            ((cursorPos[0], cursorPos[1]), (cursorPos[0] + 20, cursorPos[1] + 20)),  # The cursor
            ((1134, 66), (1335, 269))   # The minimap
        ]
    #tileImages = [[0 for x in range(tileWidth - 2)] for y in range(tileHeight - 2)]
    #x_adjust = 16 - int(captureData['ScreenPosX'])%16
    #y_adjust = 16 - int(captureData['ScreenPosY'])%16
    tileImages = tileImage(img, captureData)
    saveTiles(tileImages, captureData, nogoZone)

    if endMessage is not None: print(endMessage)
