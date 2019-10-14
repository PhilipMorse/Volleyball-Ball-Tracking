import cv2
import numpy as np
import time

CREATE_TRAINING_DATA = True
VID_NAME = "vid1.mp4"

cap = cv2.VideoCapture(VID_NAME)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.dilate(opening, kernel, iterations=1)
    #erosion = cv2.erode(fgmask, kernel, iterations=2)

    _, thresh = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        if len(contours)<10:
            if CREATE_TRAINING_DATA:
                frame_n = cap.get(cv2.CAP_PROP_POS_FRAMES)
                for i,c in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(c)
                    roi = frame[y: y+h, x: x+w]
                    filename = 'data/' + VID_NAME + '-' + str(frame_n) + '-' + str(i) + '.png'
                    cv2.imwrite(filename, roi)
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('original', frame)
    cv2.imshow('opened', opening)

    k = cv2.waitKey(0)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
