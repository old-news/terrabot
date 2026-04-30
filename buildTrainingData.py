import os
import json
import cv2
import random
from snipCapture import snipImageAndSaveClassified
from janitor import removeDuplicates, clampMaxFiles, clearTrainingData
import time
import shutil

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
    start = time.perf_counter()
    print("Clearing old training data...")
    clearTrainingData()
    shutil.rmtree('./validation/tile')
    os.mkdir('./validation/tile')
    nameIDs = [int(name.split('.')[0]) for name in os.listdir(imageFramePath)]
    lastNameID = max(nameIDs)
    random.shuffle(nameIDs)
    for i, nameID in enumerate(nameIDs):
        buildTileIDPair(nameID, message=f"Finished classifying frame {nameID}/{lastNameID} \t(approximately {len(nameIDs) - i} left)")
    clampMaxFiles(maxFiles=3000)
    print('Removing duplicates...')
    removeDuplicates()
    clampMaxFiles(maxFiles=2500)
    print(f'Built training data in {(time.perf_counter() - start) / 60:.2f} minutes')
    clampMaxFiles(directory='./validation/tile', maxFiles=500)
    trainingPaths = os.listdir('./training/tile')
    for path in os.listdir('./validation/tile'):
        if path not in trainingPaths:
            shutil.rmtree(f'./training/tile/{path}')
    valPaths = os.listdir('./validation/tile')
    for path in os.listdir('./validation/tile'):
        if path not in valPaths:
            shutil.rmtree(f'./training/tile/{path}')


buildTileTrainingData()
