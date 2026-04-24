import cv2
import json
import time
import os
from snipCapture import snipImageAndSaveClassified
from janitor import clampMaxFiles, removeDuplicates
from buildTrainingData import buildTileIDPair
import random


imageFramePath = './captureData/imageFrames'
dataFramePath = './captureData/dataFrames'
usedIDs = set()

while True:
    time.sleep(0.1)
    nameIDs = [name.split('.')[0] for name in os.listdir(imageFramePath)]
    newIDs = set(nameIDs) - usedIDs
    for newID in newIDs:
        buildTileIDPair(newID)
        usedIDs.add(newID)
    if len(newIDs) > 0 and random.random() > 2/3:
        removeDuplicates()
        clampMaxFiles(500)
