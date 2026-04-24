import os, shutil, sys, json
import numpy as np
from PIL import Image
import random


classTilePath = './training/tile'
imageFramePath = './captureData/imageFrames'
dataFramePath = './captureData/dataFrames'

def clampMaxFiles(maxFiles=None):
    if maxFiles is None: maxFiles = 250
    for blockClass in os.listdir(classTilePath):
        rootPath = f'{classTilePath}/{blockClass}'
        if len(os.listdir(rootPath)) > maxFiles:
            print(f'Clamping {rootPath} from {len(os.listdir(rootPath))} to {maxFiles} files')
            paths = os.listdir(rootPath)
            iterations = len(os.listdir(rootPath)) - maxFiles
            for i in range(iterations):
                toRemove = random.randint(0, len(paths) - 1)
                fullPath = f'{rootPath}/{paths[toRemove]}'
                paths.pop(toRemove)
                try:
                    os.remove(fullPath)
                except:
                    pass

def removeDuplicates():
    for blockClass in os.listdir(classTilePath):
        images = dict()
        rootPath = f'{classTilePath}/{blockClass}'
        print(f'Removing duplicates from {blockClass} ({len(os.listdir(rootPath))} files)')
        for path in os.listdir(rootPath):
            fullPath = f'{classTilePath}/{blockClass}/{path}'
            images[json.dumps(np.array(Image.open(fullPath)).tolist())] = fullPath
        goodPaths = [path for path in images.values()]
        allPaths = [f'{classTilePath}/{blockClass}/{path}' for path in os.listdir(rootPath)]
        badPaths = [path for path in allPaths if path not in goodPaths]
        for path in badPaths:
            os.remove(path)
        continue

def savermtree(path):
    shutil.rmtree(path)
    os.mkdir(path)

def clearCaptureData():
    savermtree(dataFramePath)
    savermtree(imageFramePath)

def clearTrainingData():
    savermtree('./training/tile')


if __name__ == '__main__':
    helpString = "That is not a valid command\nUsage:\n\tclear - clears all capture data\n\ttclear - clears all training data\n\tclean - clamps and removes duplicates from training data\n\treset - clears all training and capture data, and deletes the CNN"
    if len(sys.argv) == 1:
        print(helpString)
        exit(0)
    if sys.argv[1] == 'reset':
        clearCaptureData()
        clearTrainingData()
        try:
            os.remove('img2Tile.cnn')
        except:
            pass
    elif sys.argv[1] == 'clean':
        clampMaxFiles()
        removeDuplicates()
    elif sys.argv[1] == 'clear':
        clearCaptureData()
    elif sys.argv[1] == 'tclear':
        clearTrainingData()
    else:
        print(helpString)
