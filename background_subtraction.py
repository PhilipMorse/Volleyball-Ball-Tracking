import cv2

CREATE_TRAINING_DATA = True
VID_NAME = "vid3"
VID_FORMAT = ".mp4"

cap = cv2.VideoCapture("vids/" + VID_NAME + VID_FORMAT)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

while True:
    ret, frame = cap.read()
    frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.dilate(opening, kernel, iterations=1)

    _, thresh = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)
    if frame_n > 1:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if len(contours) < 10:
                if CREATE_TRAINING_DATA:
                    for i, c in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(c)
                        if w>10:
                            if h>10:
                                roi = frame[y: y + h, x: x + w]
                                filename = 'data/' + VID_NAME + '-' + str(frame_n) + '-' + str(i) + '.png'
                                cv2.imwrite(filename, roi)
                c = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('original', frame)
    cv2.imshow('opened', opening)

    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
