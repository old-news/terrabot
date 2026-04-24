import os
import json
import cv2
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
    nameIDs = sorted([int(name.split('.')[0]) for name in os.listdir(imageFramePath)])
    lastNameID = max(nameIDs)
    for nameID in nameIDs:
        buildTileIDPair(nameID, message=f"Finished classifying frame {nameID}/{lastNameID}")
    clampMaxFiles(500)
    removeDuplicates()
    clampMaxFiles()

buildTileTrainingData()
