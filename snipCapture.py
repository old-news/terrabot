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
    elif numID in [6, 7, 8, 9, 22, 37, 56, 58, 107, 108, 111, 166, 167, 168, 169, 204, 211, 221, 222, 223, 408]:
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
    elif numID in [185, 186, 187, 647, 648, 649, 650, 693, 694, 704, 705, ]:
        return 'rubble'
    else:
        return 'other'


def classifyImage(nparray: np.ndarray, tileInfo) -> str:
    # potentialClasses = []   # A tile might have water and a wall, so we would want to randomly decide which one so the AI learns for both cases
    # ABOVE IS DEPRECATED
    # Return in this order: unknown, tile, liquid, wall, air
    lighting = tileInfo['lighting']
    brightness = sum(lighting) / 3 / 255
    isDark = brightness < 0.1   # Tile is too dark for a human to distinguish
    tileID = tileInfo['type']
    if tileInfo['hasTile'] is not False:
        if isDark: return 'tile/unknown'
        if tileID in [3, 4, 5, 13, 24, 28, 50, 61, 82, 83, 84, 91, 105, 110, 135, 137, 144, 178, 201, 209, 215, 235, 239, 254, 323, 376, 419, 420, 423, 429, 443, 583, 584, 585, 586, 587, 588, 589, 596, 597, 616, 634, 653, 663]:
            # Tile has important semantically meaningful subID
            # For example, the tileID could be for books, with one subID being the Water Bold (ID=50)
            # Or, one of the grass types could be a mushroom rather than just decorative (ID=3)
            tilefX = tileInfo['fX']
            tilefY = tileInfo['fY']
            return f'tile/{tileID}_{tilefX}_{tilefY}'
        else:
            return f'tile/{tileID}'
    if tileInfo['liquidAmount'] != 0 and (tileInfo['isSolid'] is True or tileInfo['isActuated'] is True):
        # Liquid
        if isDark: return 'liquid/unknown'
        liquidType = tileInfo['liquidType']
        # liquidAmount = tileInfo['liquidAmount']
        return f'liquid/{liquidType}'
    wallID = tileInfo['wallType']
    if wallID != 0:
        if isDark: return 'tile/unknown'
        return f'wall/{wallID}'
    if isDark: return 'air/unknown'
    return 'air/0'

    if tileID in [4, 5, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 26, 28, 31, 33, 34, 42]:
        # Tile type is made of different materials
        return f'tile_{tileID}_{tilefX}_{tilefY}'
    if tileID in [5, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 26, 27, 28, 31, 34, 35, 36]:
        # Sprite spans multiple tiles
        return f'tile_{tileID}_{tilefX}_{tilefY}'
    if tileID in [35]:
        # Sprite changes animations
        return f'tile_{tileID}_{tilefX}_{tilefY}'
    if tileID in []:
        # Sprite meaningfully spans multiple tiles
        return f'tile_{tileID}_{tilefX}_{tilefY}'


def saveClassifyMiddleSnip(img: np.ndarray, captureData: dict) -> None:
    #img[284:484, 583:783]
    middleImg = img[284:484, 583:783]
    x_offset = int(15 - (captureData['ScreenPosX'] % 16))
    y_offset = int(15 - (captureData['ScreenPosY'] % 16))
    xPath = f'{trainingPath}/offset/x_{x_offset}/'
    yPath = f'{trainingPath}/offset/y_{y_offset}/'
    cv2.imwrite(f'{xPath}/{len(os.listdir(xPath))}.png', middleImg)
    cv2.imwrite(f'{yPath}/{len(os.listdir(yPath))}.png', middleImg)


def tileImage(img: np.ndarray, nogoZone=None) -> np.ndarray:
    height, width, channels = img.shape
    tileWidth, tileHeight = floor(width / 16) - 1, floor(height / 16) - 1
    tileImages = [[0 for x in range(tileWidth - 2)] for y in range(tileHeight - 2)]
    for x in range(tileWidth - 2):
        for y in range(tileHeight - 2):
            isInNogoZone = False
            if nogoZone is not None:
                corners = ((16*x, 16*y), (16*(x+3), 16*y), (16*x, 16*(y+3)), (16*(x+3), 16*(y+3)))
                for zone in nogoZone:
                    if isInNogoZone: break
                    for corner in corners:
                        if corner[0] > zone[0][0] and corner[0] < zone[1][0] and corner[1] > zone[0][1] and corner[1] < zone[1][1]:
                            isInNogoZone = True
                            break
            if isInNogoZone: continue
            #newtilePixelArray = img[16*y + y_adjust:16*(y+1) + y_adjust, 16*x + x_adjust:16*(x+1) + x_adjust, :]
            newtilePixelArray = img[16*y:16*(y+3), 16*x:16*(x+3), :]
            tileImages[y][x] = newtilePixelArray
    return tileImages


def saveTiles(tileImages: np.ndarray, tileData: dict) -> None:
    classifiedImageCounts = dict()
    # classifiedImageCounts is used because os.listdir takes a lot of time
    for mainPath in os.listdir(trainingPath):
        for path in os.listdir(f'{trainingPath}/{mainPath}'):
            classifiedImageCounts[f'{mainPath}/{path}'] = len(os.listdir(f'{trainingPath}/{mainPath}/{path}'))

    for y, imageRow in enumerate(tileImages):
        for x, image in enumerate(imageRow):
            #print(imageRow)
            if not isinstance(image, np.ndarray): continue
            # if np.mean(image) < 25: continue    # Image is too dark to tell what it is
            # ABOVE IS DEPRECATED: using tile lighting from tModLoader to determine if tile is too dark
            tileInfo = tileData[x+1][y+1]
            classified = classifyImage(image, tileInfo)
            path = f'{trainingPath}/{classified}'
            imgAlreadyInSet = False
            nameID = classifiedImageCounts.get(classified)
            if nameID is None:
                os.mkdir(path)
                nameID = 0
            if imgAlreadyInSet: continue
            if random.random() > 1 / (math.e ** ((nameID - 500) / 500)):
                # Do not store a crap ton of images for common dataIDs (like dirt)
                continue
            classifiedImageCounts[classified] = nameID + 1
            cv2.imwrite(f'{path}/{nameID}.png', image)


def snipImageAndSaveClassified(img: np.ndarray, captureData, endMessage=None, nogoZone=None) -> None:
    saveClassifyMiddleSnip(img, captureData)
    if nogoZone is None:
        # This nogoZone is formatted for a UI scale of 80%
        cursorPos = captureData['CursorPos']
        nogoZone = [
            ((1135, 13), (1330, 63)),   # Health
            ((1335, 16), (1360, 197)),  # Mana
            ((17, 0), (357, 56)),       # Hotbar
            ((25, 60), (348, 134)),     # Two rows of buffs. Four rows goes to y=211
            ((660, 405), (707, 465)),    # The player
            ((cursorPos[0], cursorPos[1]), (cursorPos[0] + 20, cursorPos[1] + 20))  # The cursor
        ]
    #tileImages = [[0 for x in range(tileWidth - 2)] for y in range(tileHeight - 2)]
    #x_adjust = 16 - int(captureData['ScreenPosX'])%16
    #y_adjust = 16 - int(captureData['ScreenPosY'])%16
    tileImages = tileImage(img, nogoZone)
    saveTiles(tileImages, captureData['TileData'])

    if endMessage is not None: print(endMessage)
