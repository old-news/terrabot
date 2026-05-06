import os, shutil, sys, json
import numpy as np
from PIL import Image
import random
import dataEnum


classTilePath = './training/tile'
trainingPath = './training'
imageFramePath = './captureData/imageFrames'
dataFramePath = './captureData/dataFrames'


def clampMaxFiles(directory=None, maxFiles=None):
    if directory is None: directory = trainingPath
    if maxFiles is None: maxFiles = 500
    filesInDir = []
    paths = os.listdir(directory)
    if len(paths) == 0:
        os.rmdir(directory)
        return
    for path in paths:
        subPath = os.path.join(directory, path)
        if os.path.isdir(subPath):
            clampMaxFiles(subPath, maxFiles)
        else:
            filesInDir.append(subPath)
    if len(filesInDir) <= maxFiles: return
    print(f'Clamping {directory}: {len(filesInDir)} -> {maxFiles} files')
    for i in range(len(filesInDir) - maxFiles):
        path = random.choice(filesInDir)
        os.remove(path)
        filesInDir.remove(path)
    return
    for datatype in os.listdir(trainingPath):
        for dataclass in os.listdir(f'{trainingPath}/{datatype}'):
            rootPath = f'{trainingPath}/{datatype}/{dataclass}'
            if len(os.listdir(rootPath)) > maxFiles:
                print(f'Clamping {rootPath}: {len(os.listdir(rootPath))} -> {maxFiles} files')
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


def removeDuplicates(directory=None):
    if directory is None: directory = trainingPath
    files = []
    for path in os.listdir(directory):
        subPath = os.path.join(directory, path)
        if os.path.isdir(subPath):
            removeDuplicates(subPath)
        else:
            files.append(subPath)
    hashes = dict()
    for path in files:
        with open(path, 'rb') as ifile:
            hashes[ifile.read()] = path
    difference = set(files) - set(hashes.values())
    if len(difference) > 0:
        print(f'Removing duplicates from {directory}: {len(files)} -> {len(hashes.values())} files')
    for path in difference:
        os.remove(path)
    return
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


def removeEmptyDirs(directory=None):
    if directory is None: directory = trainingPath
    paths = os.listdir(directory)
    if len(paths) == 0:
        os.rmdir(directory)
        return
    for path in paths:
        if not os.path.isdir(os.path.join(directory, path)): continue
        removeEmptyDirs(directory)


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
    os.mkdir('./training/offset_x')
    os.mkdir('./training/offset_y')
    for i in range(16):
        os.mkdir(f'./training/offset_x/{i}')
        os.mkdir(f'./training/offset_y/{i}')
    for path in dataEnum.tileCategories:
        os.mkdir(f'./training/tile/{path}')


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
