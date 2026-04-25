import os, shutil, sys, json
import numpy as np
from PIL import Image
import random


classTilePath = './training/tile'
trainingPath = './training'
imageFramePath = './captureData/imageFrames'
dataFramePath = './captureData/dataFrames'

def clampMaxFiles(maxFiles=None):
    if maxFiles is None: maxFiles = 250
    for datatype in os.listdir(trainingPath):
        for dataclass in os.listdir(f'{trainingPath}/{datatype}'):
            rootPath = f'{trainingPath}/{datatype}/{dataclass}'
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
    for datatype in os.listdir(trainingPath):
        for dataclass in os.listdir(f'{trainingPath}/{datatype}'):
            images = dict()
            rootPath = f'{trainingPath}/{datatype}/{dataclass}'
            print(f'Removing duplicates from {rootPath} ({len(os.listdir(rootPath))} files)')
            for path in os.listdir(rootPath):
                fullPath = f'{trainingPath}/{datatype}/{dataclass}/{path}'
                images[json.dumps(np.array(Image.open(fullPath)).tolist())] = fullPath
            goodPaths = [path for path in images.values()]
            allPaths = [f'{trainingPath}/{datatype}/{dataclass}/{path}' for path in os.listdir(rootPath)]
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
    savermtree('./training/')
    os.mkdir('./training/tile')
    os.mkdir('./training/wall')
    os.mkdir('./training/air')
    os.mkdir('./training/liquid')
    os.mkdir('./training/offset')
    for i in range(16):
        os.mkdir(f'./training/offset/x_{i}')
        os.mkdir(f'./training/offset/y_{i}')


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
