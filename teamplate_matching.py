import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

video = cv2.VideoCapture("vid2.mp4")


template = cv2.imread("ball.png", 0)
template = template[10:40, 10:40]
w,h = template.shape[::-1]

method = eval('cv2.TM_CCOEFF_NORMED')






while True:
    _, frame = video.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.show()

    key = cv2.waitKey(0)
    if key==27:
        break

#video.release()
#cv2.destroyAllWindows()