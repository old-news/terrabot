import pyautogui
import cv2
import numpy as np
import time
from PIL import Image, ImageGrab
import json

time.sleep(1)
resolution = (1366, 768)
codec = cv2.VideoWriter_fourcc(*'mp4v')
filename = "./captureData/videoCaptures/capture.mp4"
fps = 40.0
out = cv2.VideoWriter(filename, codec, fps, resolution)

fps = []
frameCounter = 0
stamps = []
start = time.time()

while frameCounter < 800:
    img = pyautogui.screenshot()
    stamp = time.time()
    stamps.append(stamp)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)

    elapsed = time.time() - start
    fps.append(1 / elapsed)
    start = time.time()
    frameCounter+=1

out.release()

with open('./captureData/videoCaptures/timeStamps.json', 'w+') as ofile:
    ofile.write(json.dumps(stamps))
print('Average FPS:', sum(fps) / len(fps))
