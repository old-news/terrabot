from PIL import Image
import json
from math import floor, ceil
import numpy as np
import os
import cv2
from functools import lru_cache
from pathlib import Path
import random


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
    tileID = tileInfo['type']
    if tileID == 0 and tileInfo['hasTile'] == False:
        # Air or liquid
        tileID = -1
    tilefX = tileInfo['fX']
    tilefY = tileInfo['fY']
    slope = tileInfo['blockType']
    return f'{tileID}_{tilefX}_{tilefY}'


def removeDuplicates():
    for blockClass in os.listdir('./captureData/classified_data'):
        print(f'Removing duplicates from {blockClass}')
        images = dict()
        rootPath = f'./captureData/classified_data/{blockClass}'
        for path in os.listdir(rootPath):
            fullPath = f'./captureData/classified_data/{blockClass}/{path}'
            images[json.dumps(np.array(Image.open(fullPath)).tolist())] = fullPath
        goodPaths = [path for path in images.values()]
        allPaths = [f'./captureData/classified_data/{blockClass}/{path}' for path in os.listdir(rootPath)]
        badPaths = [path for path in allPaths if path not in goodPaths]
        for path in badPaths:
            os.remove(path)
        continue
        for i, arr1 in enumerate(nparrays):
            for j, arr2 in enumerate(nparrays):
                if i == j: continue
                if areNpArraysSimilar(data1[0], data2[0]):
                    images.pop(j)
                    os.remove(data2[1])
                    j-=1


def clampMaxFiles():
    for blockClass in os.listdir('./captureData/classified_data'):
        rootPath = f'./captureData/classified_data/{blockClass}'
        if len(os.listdir(rootPath)) > 100:
            paths = os.listdir(rootPath)
            iterations = len(os.listdir(rootPath)) - 100
            for i in range(iterations):
                toRemove = random.randint(0, len(paths) - 1)
                fullPath = f'{rootPath}/{paths[toRemove]}'
                paths.pop(toRemove)
                try:
                    os.remove(fullPath)
                    print(f"Removed {fullPath}")
                except:
                    print(f"tried to remove {fullPath}")


def snipImageAndSaveClassified(img, captureData, endMessage=None):
    #img = Image.open('captureData/newWorld.png')
    #captureData = None
    #with open('captureData/capture0.json') as infile:
        #captureData = json.loads(infile.read())
    pixels = np.asarray(img)
    width, height = img.size
    tileWidth, tileHeight = floor(width / 16) - 1, floor(height / 16) - 1
    tileImages = [[0 for x in range(tileWidth)] for y in range(tileHeight)]
    tileData = captureData['TileData']
    x_adjust = 16 - int(captureData['ScreenPosX'])%16
    y_adjust = 16 - int(captureData['ScreenPosY'])%16
    for x in range(tileWidth):
        for y in range(tileHeight):
            newtilePixelArray = pixels[16*y + y_adjust:16*(y+1) + y_adjust, 16*x + x_adjust:16*(x+1) + x_adjust, :]
            tileImages[y][x] = newtilePixelArray

    classifiedImageCounts = dict()
    for path in os.listdir('./captureData/classified_data'):
        classifiedImageCounts[path] = len(os.listdir(f'./captureData/classified_data/{path}'))

    for y, imageRow in enumerate(tileImages):
        for x, image in enumerate(imageRow):
            if np.mean(image) < 25: continue    # Image is too dark to tell what it is
            tileInfo = tileData[x][y]
            classified = classifyImage(image, tileInfo)
            path = f'captureData/classified_data/{classified}/'
            imgAlreadyInSet = False 
            nameID = classifiedImageCounts.get(classified)
            if nameID is None:
                os.mkdir(path)
                nameID = 0
            classifiedImageCounts[classified] = nameID + 1
            #try:
                #classifiedImages = os.listdir(path)
                #pathMax = -1
                #for cImagePath in classifiedImages:
                    #numID = int(cImagePath.split('.')[0])
                    #if numID > pathMax:
                        #pathMax = numID
                    #cimage = getImageArray(path + cImagePath)
                    #if areImagesSimilar(image, cimage):
                        #imgAlreadyInSet = True
                        #break;
                #pathMax += 1
            #except:
                #os.mkdir(path)
            if imgAlreadyInSet: continue
            Image.fromarray(image).save(f'captureData/classified_data/{classified}/{nameID}.png')
    if endMessage is not None: print(endMessage)
