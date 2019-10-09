import cv2
import numpy as np

import time

pos1 = [0, 0]
frame = None
count = 0

video = cv2.VideoCapture("vid2.mp4")


def click_event(event, x, y, flags, param):
    global count, pos1
    if event == cv2.EVENT_LBUTTONDOWN:
        if count % 2 == 0:
            pos1[0] = x
            pos1[1] = y
            count += 1
        else:
            get_roi(pos1,[x,y])
            count+=1


def get_roi(pos, pos2):
    roi = frame[pos[1]: pos2[1], pos[0]: pos2[0]]
    filename = 'ref/ball-'+ str(int(time.time())) + '.png'
    print(filename)
    cv2.imwrite(filename,roi)



while True:
    _, frame = video.read()

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback('Frame', click_event)

    key = cv2.waitKey(0)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
