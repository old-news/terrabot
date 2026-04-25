import mss
import numpy as np
import cv2
import os
import time
import json
import threading
from snipCapture import snipImageAndSaveClassified
import queue

def matchFrame(name, imagesTimestamps):
    nameID = name.split('.')[0]
    dataFrame = None
    while dataFrame == None:
        try:
            with open(f'{dataFrameDir}{name}', 'r') as ifile:
                # print(f'Reading {dataFrameDir}{name}')
                dataFrame = json.loads(ifile.read())
        except:
            time.sleep(1)   # This gives the Mod time to finish writing the JSON file
    timestamp = dataFrame['Timestamp']
    image_rgb = None
    for image, stamp in imagesTimestamps:
        if stamp < timestamp: continue
        if stamp - timestamp > 1/120:
            print(f"No tick-matching frame capture for dataframe {nameID}")
            break; # Skip if the image capture and dataFrame capture are on different ticks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(f'{imageFrameDir}{nameID}.png', image_rgb)
        print(f'Successfully matched dataframe {nameID}')
        break;
    if image_rgb is None:
        print(f"Could not find valid timestamp for dataframe {nameID}")
        return

with mss.MSS() as sct:
    print("Setting up...")
    monitor = sct.monitors[1]
    imagesTimestamps = []
    dataFrameDir = './captureData/dataFrames/'
    imageFrameDir = './captureData/imageFrames/'
    matchedDataFrames = set(os.listdir(dataFrameDir))

    start = time.perf_counter()
    fps = []
    counter = 0
    print("Recording")
    while True:
        counter+=1
        captureTime = time.time()
        mss_img = sct.grab(monitor)
        img = np.array(mss_img)
        imagesTimestamps.append((img, captureTime))
        if len(imagesTimestamps) > 60: imagesTimestamps.pop(0)
        dataFramePaths = set(os.listdir(dataFrameDir))
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        fps.append(1 / elapsed)
        if len(fps) > 720:
            fps.pop(0)
        if len(imagesTimestamps) > 720:
            imagesTimestamps.pop(0)
        if counter > 300:
            print("Average fps:", sum(fps) / len(fps))
            counter = 0
        if matchedDataFrames == dataFramePaths: continue
        newDataFrames = dataFramePaths.difference(matchedDataFrames)
        for name in newDataFrames:
            matchedDataFrames.add(name)
            # matchFrame(name, imagesTimestamps)
            threading.Thread(target=matchFrame, args=(name, imagesTimestamps[:]), daemon=True).start()
