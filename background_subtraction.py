import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

USE_IMAGE_CLASSIFICATION_MODEL = False  # Determine whether to used image classification model
MODEL = "0.9818182-1571182058"  # Model name to be used (.h5 format)
CREATE_TRAINING_DATA = False  # Used to create training set in data/
VID_NAME = "vid1"  # Video file found in vids/
VID_FORMAT = ".mp4"  # Video format
KERNEL_RESOLUTION = 200  # Higher = More objects detected in frame

cap = cv2.VideoCapture("vids/" + VID_NAME + VID_FORMAT)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
kernel_size = int(cap_width / KERNEL_RESOLUTION)
print(cap_width, kernel_size)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

model = None

if USE_IMAGE_CLASSIFICATION_MODEL:
    model = load_model('models/' + MODEL + '.h5')


def classify_contours(contour_list, target_frame):
    probs = []
    for c in contour_list:
        x, y, w, h = cv2.boundingRect(c)
        roi = target_frame[y: y + h, x: x + w]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (28, 28))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        probability = model.predict_proba(img)
        probs.append(probability[0][1])
        cv2.putText(frame, str(probability[0][1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return probs


while True:
    ret, frame = cap.read()
    frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.dilate(opening, kernel, iterations=1)

    _, thresh = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)
    if frame_n > 1:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if len(contours) < 100:
                if CREATE_TRAINING_DATA:
                    for i, c in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(c)
                        if w > 10:
                            if h > 10:
                                roi = frame[y: y + h, x: x + w]
                                filename = 'data/' + VID_NAME + '-' + str(frame_n) + '-' + str(i) + '.png'
                                cv2.imwrite(filename, roi)
                if USE_IMAGE_CLASSIFICATION_MODEL:
                    probabilities = classify_contours(contours, frame)

                    # TODO: Change this naive ball finding
                    max_index = probabilities.index(max(probabilities))
                    x, y, w, h = cv2.boundingRect(contours[max_index])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    probabilities = [1 for i in range(len(contours))]
                    c = max(contours, key=cv2.contourArea)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print(probabilities)


    cv2.imshow('original', frame)
    cv2.imshow('opened', opening)

    k = cv2.waitKey(0)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
