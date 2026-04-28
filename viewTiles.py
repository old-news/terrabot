from snipCapture import tileImage, classifyImage, parseTileData
import json
import cv2
import time

if __name__ == '__main__':
    index = 8
    image = cv2.imread(f'./captureData/imageFrames/{index}.png')
    captureData = None
    with open(f'./captureData/dataFrames/{index}.json', 'r') as ifile:
        captureData = json.loads(ifile.read())
    tiles = tileImage(image, captureData)
    xmod = captureData['ScreenPosX'] % 16
    ymod = captureData['ScreenPosY'] % 16
    xadjust = 1  #1 + int(xmod >= 8)  # 1 + int(xmod >= 8)
    yadjust = 0  #0 + int(ymod >= 8)
    x = 1
    y = 27
    tileInfo = parseTileData(captureData['TileData'])[x+xadjust][y+yadjust]
    print(xmod, ymod)
    print(tileInfo)
    print(classifyImage(image, tileInfo))
    cv2.imshow('im', tiles[y][x])
    cv2.imwrite('im.png', tiles[y][x])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
