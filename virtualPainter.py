import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

################
eraserThickness = 35
brushThickness = 15
################
folderPath = 'header'
myList = os.listdir(folderPath)

print(myList)
overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderPath}/{impath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
# making sure the height and width are of size of webcam.
cap.set(2, 796)
cap.set(2, 1280)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)
while True:
    # 1. import image
    success, img = cap.read()
    # flipping horizontally when we move onto right we will draw onto left and vice versa.

    img = cv2.flip(img, 1)
    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #print(lmList)
        # TIP OF INDEX AND MIDDLE FINGER
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which finger are up we want to paint when only index finger is up and when two fingers are up we
        # can move around

        fingers = detector.fingersUp()
        # print(fingers)
        # 4. If Selection mode = two fingers are up.
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            print('Selection Mode')
            if y1 < 55:
                # checking for the click
                if 100 < x1 < 120:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 200 < x1 < 300:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 300 < x1 < 400:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 400 < x1 < 500:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        # 5. if Drawing Mode = Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 12, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    img[:55, :] = header
    # little bit transparent
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('Image', img)
    cv2.imshow('Canvas', imgCanvas)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # cap.release()
    # cv2.destroyAllWindows()
