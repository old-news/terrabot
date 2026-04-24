import os
import json
import cv2
from snipCapture import snipImageAndSaveClassified, removeDuplicates, clampMaxFiles 
from PIL import Image
import threading
import time

dataPath = './captureData/tileCaptures/'
videoPath = os.getcwd() + f'/captureData/videoCaptures/capture.mp4'
trainingFrames = []
dataFrames = []
videoFrames = []
print(os.getcwd())

def extractVideoFrames():
    global videoPath
    video = cv2.VideoCapture(videoPath)
    print(videoPath)
    print("Extracting video frames")
    frameNum, success = 0, True
    while success:
        time.sleep(0.01)
        success, image = video.read()
        if not success: break;
        frameNum += 1
        if (frameNum - videoIndexOffset) % 60 != 0: continue
        convertedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pilImg = Image.fromarray(convertedImage)
        videoFrames.append(pilImg);
    print(f"Extracted {frameNum} frames from video")

def selectVideoFrames(dataFrames):
    video = cv2.VideoCapture('./captureData/videoCaptures/capture.mp4')
    timeStamps = []
    with open('./captureData/videoCaptures/timeStamps.json', 'r') as infile:
        timeStamps = json.loads(infile.read())
    videoFrameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('videoFarmCount:', videoFrameCount)
    videoStart = timeStamps[0]
    selectedFrames = []
    if videoStart > dataFrames[0]['Timestamp']: return
    for dataFrame in dataFrames:
        offsetList = [stamp - dataFrame['Timestamp'] for stamp in timeStamps]
        offsetList = [val if val > 0 else 99999999999999999 for val in offsetList]
        bestIndex = offsetList.index(min(offsetList))
        if bestIndex >= videoFrameCount or offsetList[bestIndex] > 999999: break;
        selectedFrames.append(bestIndex)
    frameNum, success = 0, True
    while success:
        time.sleep(0.002)
        succes, image = video.read()
        if not succes: break;
        if frameNum in selectedFrames:
            convertedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pilImg = Image.fromarray(convertedImage)
            path = f'./captureData/imageCaptures/{selectedFrames.index(frameNum)}.png'
            pilImg.save(path)
            print(f'Saved {path}')
        frameNum += 1
    imagePaths = [f'./captureData/imageCaptures/{i}.png' for i in range(len(selectedFrames))]
    print(imagePaths)
    return imagePaths


    if videoStart < dataFrames[0]['Timestamp']:
        # The video started before the tile recording
        msOffset = dataFrames[0]['Timestamp'] - videoStart
        videoFrameOffset = int(msOffset / (60 / 1000))
        print(videoFrameOffset, videoFrameCount, 60)
        selected = [i for i in range(videoFrameOffset, videoFrameCount, 60)]
        frameNum, success = 0, True
        while success:
            time.sleep(0.01)
            succes, image = video.read()
            if not succes: break;
            if frameNum in selected:
                convertedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(convertedImage)
                pilImg.save(f'./captureData/imageCaptures/{selected.index(frameNum)}.png')
            frameNum += 1
        imagePaths = [f'./captureData/imageCaptures/{i}.png' for i in range(len(selected))]
        print(imagePaths)
        return imagePaths

dataFrameFirst = None
with open('./captureData/tileCaptures/capture0.json', 'r') as infile:
    dataFrameFirst = json.loads(infile.read())


for path in os.listdir(dataPath):
    fullPath = f'{dataPath}{path}'
    print(f'{dataPath}{path}')
    with open(fullPath, 'r') as infile:
        dataFrame = json.loads(infile.read())
        dataFrames.append(dataFrame)

dataFrames.sort(key=lambda x: x['Timestamp'])
videoFramePaths = selectVideoFrames(dataFrames)
print('dataframes:', len(dataFrames), 'videoframes:', len(videoFramePaths))
for i, frame in enumerate(dataFrames):
    print(frame['Timestamp'])
    if i >= len(videoFramePaths): break;
    videoFrame = Image.open(videoFramePaths[i])
    snipImageAndSaveClassified(videoFrame, frame, f"Finished classifying frame {i} (up to frame {min(len(dataFrames), len(videoFramePaths)) - 1})")

print("Removing duplicate images...")
removeDuplicates()
print("Duplicates removed")
clampMaxFiles()
exit(0)




diff = videoStart - dataFrames[0]['Timestamp']
offset = abs(diff)
print(f'start diff: {diff}')
tileIndexOffset = 0
diff = videoStart - dataFrames[1]['Timestamp']
print(f'brand new diff: {diff}')
while abs(diff) < offset and diff < 0:
    tileIndexOffset+=1
    offset = abs(diff)
    diff = videoStart - dataFrames[tileIndexOffset]['Timestamp']
    print(f'new diff: {diff}')

videoIndexOffset = int(60 * (dataFrames[tileIndexOffset]['Timestamp'] - videoStart))
print('diff:', dataFrames[tileIndexOffset]['Timestamp'] - videoStart)
maxFrames = min(len(dataFrames) - tileIndexOffset, int((len(videoFrames) - videoIndexOffset) / 60))
extractVideoFrames()
threads = []
loops = 0
for i in range(tileIndexOffset, maxFrames):
    # Iterate over dataFrames and videoFrames
    # dataFrames[i] matches up with videoFrames[videoIndexOffset + 60*i]
    # filteredFrames = [[] for i in range(dataFrames[i].TileData)]
    # for x in range(len(dataFrames[i].TileData[0])):
        # for y in range(len(dataFrames[i].TileData)):
            # tileInfo = dataFrames[i].TileData[x][y]
            # strippedInfo = [tileInfo.type, tileInfo.hasTile and tileInfo.isSolid]
            # filteredFrames.append(strippedInfo)
    # The image is the corresponding image
    # The answer is the correct id's of the tiles, as well as if they're a solid tile or not
    corrImg = videoFrames[loops]
    loops+=1
    corrTileFrame = dataFrames[i] #filteredFrames
    #newThread = threading.Thread(target=snipImageAndSaveClassified, args=(corrImg, corrTileFrame, f"Finished classifying frame {i} (up to frame {maxFrames})"))
    #newThread.start()
    #threads.append(newThread)
    snipImageAndSaveClassified(corrImg, corrTileFrame, f"Finished classifying frame {i} (up to frame {maxFrames})")

print("Removing duplicate images...")
removeDuplicates()
print("Duplicates removed")
clampMaxFiles()
exit(0)
loops = 0
for i in range(tileIndexOffset, maxFrames):
    threads[loops].join()
    loops+=1
