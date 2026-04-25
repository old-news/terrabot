import os
import json
import cv2
import random
from snipCapture import snipImageAndSaveClassified
from janitor import removeDuplicates, clampMaxFiles 

dataFramePath = './captureData/dataFrames'
imageFramePath = './captureData/imageFrames'
print(os.getcwd())

def buildTileIDPair(nameID, message=None):
    dataFrame = None
    with open(f'{dataFramePath}/{nameID}.json', 'r') as ifile:
        dataFrame = json.loads(ifile.read())
    image = cv2.imread(f'{imageFramePath}/{nameID}.png')
    snipImageAndSaveClassified(image, dataFrame, message)


def buildTileTrainingData():
    nameIDs = [int(name.split('.')[0]) for name in os.listdir(imageFramePath)]
    lastNameID = max(nameIDs)
    random.shuffle(nameIDs)
    for i, nameID in enumerate(nameIDs):
        buildTileIDPair(nameID, message=f"Finished classifying frame {nameID}/{lastNameID} (approximately {len(nameIDs) - i} left)")
    clampMaxFiles(750)
    removeDuplicates()
    clampMaxFiles(500)

buildTileTrainingData()
