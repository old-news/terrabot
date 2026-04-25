from PIL import Image
import json
from math import floor, ceil
import numpy as np
import os
import cv2
from functools import lru_cache
from pathlib import Path
import random
import time
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


def classifyImage(nparray, tileInfo):
    # potentialClasses = []   # A tile might have water and a wall, so we would want to randomly decide which one so the AI learns for both cases
    # ABOVE IS DEPRECATED
    # Return in this order: unknown, tile, liquid, wall, air
    lighting = tileInfo['lighting']
    brightness = sum(lighting) / 3 / 255
    isDark = brightness < 0.1   # Tile is too dark for a human to distinguish
    tileID = tileInfo['type']
    if tileInfo['hasTile'] != False:
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
    if tileInfo['liquidAmount'] != 0 and (tileInfo['isSolid'] == False or tileInfo['isActuated'] == True):
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


def saveClassifyMiddleSnip(img, captureData):
    #img[284:484, 583:783]
    middleImg = img[284:484, 583:783]
    x_offset = int(15 - (captureData['ScreenPosX'] % 16))
    y_offset = int(15 - (captureData['ScreenPosY'] % 16))
    xPath = f'{trainingPath}/offset/x_{x_offset}/'
    yPath = f'{trainingPath}/offset/y_{y_offset}/'
    cv2.imwrite(f'{xPath}/{len(os.listdir(xPath))}.png', middleImg)
    cv2.imwrite(f'{yPath}/{len(os.listdir(yPath))}.png', middleImg)


def snipImageAndSaveClassified(img, captureData, endMessage=None, nogoZone=None):
    saveClassifyMiddleSnip(img, captureData)
    if nogoZone is None:
        # This nogoZone is formatted for a UI scale of 80%
        cursorPos = captureData['CursorPos']
        nogoZone = [
            ((1135, 13), (1330, 63)),   # Health
            ((1335, 16), (1360, 197)),  # Mana
            ((17, 0), (357, 56)),       # Hotbar
            ((25, 60), (348, 134)),     # Two rows of buffs. Four rows goes to y=211
            ((660, 405), (707,465)),    # The player
            ((cursorPos[0], cursorPos[1]), (cursorPos[0] + 20, cursorPos[1] + 20))  # The cursor
        ]
    pixels = img[:]
    height, width, channels = img.shape
    tileWidth, tileHeight = floor(width / 16) - 1, floor(height / 16) - 1
    tileImages = [[0 for x in range(int(tileWidth/3))] for y in range(int(tileHeight/3))]
    tileData = captureData['TileData']
    x_adjust = 16 - int(captureData['ScreenPosX'])%16
    y_adjust = 16 - int(captureData['ScreenPosY'])%16
    for x in range(int(tileWidth/3)):
        for y in range(int(tileHeight/3)):
            corners = ((16*x + x_adjust, 16*y + y_adjust), (16*(x+1) + x_adjust - 1, 16*y + y_adjust), (16*x + x_adjust, 16*(y+1) + y_adjust - 1), (16*(x+1) + x_adjust - 1, 16*(y+1) + y_adjust - 1))
            corners = ((48*x, 48*y), (48*(x+1), 48*y), (48*x, 48*(y+1)), (48*(x+1), 48*(y+1)))
            isInNogoZone = False
            for zone in nogoZone:
                if isInNogoZone: break;
                for corner in corners:
                    if corner[0] > zone[0][0] and corner[0] < zone[1][0] and corner[1] > zone[0][1] and corner[1] < zone[1][1]:
                        isInNogoZone = True
                        break;
            if isInNogoZone: continue
            #newtilePixelArray = pixels[16*y + y_adjust:16*(y+1) + y_adjust, 16*x + x_adjust:16*(x+1) + x_adjust, :]
            newtilePixelArray = pixels[48*y:48*(y+1), 48*x:48*(x+1), :]
            tileImages[y][x] = newtilePixelArray

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
            tileInfo = tileData[3*x][3*y]
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
    if endMessage is not None: print(endMessage)
