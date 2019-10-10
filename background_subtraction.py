import cv2
import numpy as np

cap = cv2.VideoCapture("vid2.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(fgmask, kernel, iterations=2)


    _, thresh = cv2.threshold(erosion, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('original', frame)
    cv2.imshow('eroded', erosion)

    k = cv2.waitKey(60)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()